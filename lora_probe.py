# probe_new.py

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ============================
# Configuration
# ============================
SEED = 43
PEFT_MODEL_ID = "matboz/ring-gemma-2-14-35"
BASE_MODEL_ID = "google/gemma-2-27b-it"
DATASET_PATH = "datasets/classification/mms_ring_classification_dataset.json"
CACHE_PATH = "caches/ring_cache.pt"
NUM_REGRESSION_RUNS = 100
MAX_PROMPTS_TO_PROCESS = None  # None to process all prompts

# Output folder + run id
SAVE_DIR = "regression_figs"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Binary labels
LABEL_POSITIVE = "self_description"
LABEL_NEGATIVE = "actual_behavior"

# Global layer ordering (filled after hooks are attached)
LAYER_NUMBERS: List[int] = []
DEFAULT_ADAPTER_NAME: str = "default"

# ============================
# Utilities
# ============================

def setup_environment() -> None:
    # Make sure outputs directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Safer attention path (also set outside is fine)
    os.environ.setdefault("HF_DISABLE_FLASH_ATTN", "1")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Prefer TF32 on matmul/cudnn for stability/perf
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Disable PyTorch SDPA variants (these do NOT disable HF flash-attn2)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    print(f"[DEBUG] torch={torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")
    print("[DEBUG] TF32 allowed; SDPA kernels disabled; HF_DISABLE_FLASH_ATTN =",
          os.getenv("HF_DISABLE_FLASH_ATTN", "<unset>"))


def get_layer_number_from_name(name: str) -> int:
    parts = name.split(".")
    if "layers" in parts:
        try:
            idx = parts.index("layers")
            return int(parts[idx + 1])
        except Exception:
            return -1
    return -1


# ============================
# Dataclasses
# ============================

@dataclass
class LayerScalars:
    layer_scalars: Dict[str, float]  # keys are str(layer_number)


@dataclass
class PromptScalars:
    scalars: List[LayerScalars]  # one entry per token in full prompt+response


@dataclass
class ProbeItem:
    prompt: str
    label: str
    response: str
    prompt_answer_first_token_idx: int
    lora_scalars: PromptScalars
    answer_token_scores: List[float]           # length == number of answer tokens
    prompt_kl_divs: List[float]                # length == number of full tokens (prompt+answer)


@dataclass
class AveragedRegressionMetrics:
    accuracy: float
    accuracy_std: float
    precision: float
    precision_std: float
    recall: float
    recall_std: float
    f1_score: float
    f1_score_std: float
    auc_roc: float
    auc_roc_std: float
    confusion_matrix: np.ndarray
    classification_report: str


# ============================
# Model / tokenizer loading — SINGLE MODEL
# ============================

def _sanity_forward_check(peft_model, tokenizer):
    dev = next(peft_model.parameters()).device
    ids = tokenizer("Hello", return_tensors="pt").to(dev).input_ids

    # Base (adapters off) using SAME forward as adapter model
    with peft_model.disable_adapter():
        with torch.no_grad():
            base_logits = peft_model(ids).logits
    with torch.no_grad():
        gen_logits = peft_model(ids).logits

    def _stats(name, t):
        finite = int(torch.isfinite(t).sum())
        total = int(t.numel())
        print(f"[DEBUG] {name}: shape={tuple(t.shape)} dtype={t.dtype} finite={finite}/{total} "
              f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}")
        return t

    b = _stats("SANITY base_logits", base_logits)
    g = _stats("SANITY gen_logits", gen_logits)
    mean_diff = (g - b).abs().mean().item()
    print(f"[DEBUG] SANITY mean|gen-base| = {mean_diff:.6f} (should be > 0 if adapters have any effect)")
    if not torch.isfinite(b).all() or not torch.isfinite(g).all():
        raise RuntimeError("NaN/Inf detected during sanity forward. Adjust attention/dtypes.")

def load_model_and_tokenizer():
    """Load base model in 4-bit, force eager attention, attach PEFT adapter."""
    print("Loading model and tokenizer…")

    bf16_ok = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"[DEBUG] bf16_supported={bf16_ok} -> compute_dtype={compute_dtype}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    peft_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    peft_model.eval()

    global DEFAULT_ADAPTER_NAME
    try:
        DEFAULT_ADAPTER_NAME = next(iter(peft_model.peft_config.keys()))
    except Exception:
        DEFAULT_ADAPTER_NAME = "default"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("[DEBUG] Active adapters now:", DEFAULT_ADAPTER_NAME, f"(default='{DEFAULT_ADAPTER_NAME}')")
    _sanity_forward_check(peft_model, tokenizer)
    print("Model and tokenizer loaded.")
    return peft_model, tokenizer


# ============================
# Hooking LoRA modules
# ============================

captured_scalars: Dict[str, torch.Tensor] = {}


def get_lora_hook(layer_name: str):
    def hook(module, inputs, output):
        tensor_output = output[0] if isinstance(output, tuple) else output
        scalars = tensor_output.detach().squeeze(-1)
        if scalars.dim() == 1:  # ensure batch dim
            scalars = scalars.unsqueeze(0)
        captured_scalars[layer_name] = scalars  # [B, T]
    return hook


def add_hooks_to_model(model: torch.nn.Module) -> Tuple[List[str], List[torch.utils.hooks.RemovableHandle]]:
    hook_handles = []
    lora_module_names: List[str] = []
    for name, module in model.named_modules():
        if "lora_A" in name and isinstance(module, torch.nn.Linear):
            clean_name = name.replace(".lora_A.default", "")
            lora_module_names.append(clean_name)
            handle = module.register_forward_hook(get_lora_hook(clean_name))
            hook_handles.append(handle)

    lora_module_names.sort(key=lambda n: (get_layer_number_from_name(n), n))

    global LAYER_NUMBERS
    LAYER_NUMBERS = sorted({get_layer_number_from_name(n) for n in lora_module_names if get_layer_number_from_name(n) >= 0})
    print(f"Hooked into {len(lora_module_names)} LoRA modules across {len(LAYER_NUMBERS)} layers: {LAYER_NUMBERS}")
    return lora_module_names, hook_handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


# ============================
# Token score & KL computation
# ============================

def _assert_all_finite(t: torch.Tensor, tag: str):
    finite = torch.isfinite(t)
    nfin = int(finite.sum())
    tot = int(t.numel())
    print(f"[DEBUG] {tag}: shape={tuple(t.shape)} dtype={t.dtype} | finite={nfin}/{tot}")
    if nfin != tot:
        raise RuntimeError(f"Non-finite values detected at stage: {tag}")

def compute_token_scores_and_kl(
    peft_model: PeftModel,
    input_ids: torch.Tensor,  # [1, T_prompt]
    response_ids: torch.Tensor,  # [1, T_answer]
) -> Tuple[List[float], List[float]]:
    """
    Compute (answer_token_scores, prompt_kl_divs_for_full_sequence).
    Base logits use adapters disabled in-place on the same model to avoid second model.
    """
    device = next(peft_model.parameters()).device
    full_ids = torch.cat([input_ids, response_ids], dim=-1).to(device)

    # --- Base logits (adapters OFF, same stack) ---
    with peft_model.disable_adapter():
        with torch.no_grad():
            base_logits_full = peft_model(full_ids).logits  # [1, T_full, V]
    _assert_all_finite(base_logits_full, "base_logits_full")

    # --- Adapter-enabled logits over full sequence (no grads) ---
    with torch.no_grad():
        gen_logits_full = peft_model(full_ids).logits  # [1, T_full, V]
    _assert_all_finite(gen_logits_full, "gen_logits_full")

    # Stable log_softmax (float32 for safety)
    gen_log_probs = F.log_softmax(gen_logits_full.float(), dim=-1)
    base_log_probs = F.log_softmax(base_logits_full.float(), dim=-1)

    kl_full = F.kl_div(
        gen_log_probs, base_log_probs, reduction="none", log_target=True
    ).sum(dim=-1).squeeze(0)  # [T_full]
    kl_full = torch.nan_to_num(kl_full, nan=0.0, posinf=10.0, neginf=0.0)
    print(f"[DEBUG] KL(full): mean={kl_full.mean().item():.6f}, >0 count={(kl_full>0).sum().item()}/{kl_full.numel()}")

    # --- Grad-based answer token significance (adapters enabled) ---
    peft_model.train()
    emb = peft_model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    outputs = peft_model(input_ids=full_ids)
    gen_logits = outputs.logits  # [1, T_full, V]
    _assert_all_finite(gen_logits, "gen_logits(train)")

    T_full = gen_logits.shape[1]
    T_answer = response_ids.shape[-1]
    answer_slice = slice(T_full - T_answer, T_full)

    with peft_model.disable_adapter():
        with torch.no_grad():
            base_logits_answer = peft_model(full_ids).logits[:, answer_slice, :]
    _assert_all_finite(base_logits_answer, "base_logits_answer")

    gen_logits_answer = gen_logits[:, answer_slice, :]

    gen_log_probs_answer = F.log_softmax(gen_logits_answer.float(), dim=-1)
    base_log_probs_answer = F.log_softmax(base_logits_answer.float(), dim=-1)

    kl_answer = F.kl_div(
        gen_log_probs_answer, base_log_probs_answer, reduction="none", log_target=True
    ).sum(dim=-1).squeeze(0)  # [T_answer]
    kl_answer = torch.nan_to_num(kl_answer, nan=0.0, posinf=10.0, neginf=0.0)
    print(f"[DEBUG] KL(answer) sum={kl_answer.sum().item():.6f}")

    total_kl_div = kl_answer.sum()
    total_kl_div.backward()

    if emb.weight.grad is None:
        print("[DEBUG] emb.weight.grad is None! No grad reached embeddings.")
        peft_model.eval()
        peft_model.zero_grad(set_to_none=True)
        emb.weight.requires_grad_(False)
        return [0.0] * T_answer, kl_full.cpu().numpy().tolist()

    token_grads = emb.weight.grad[full_ids.squeeze(0)]  # [T_full, D]
    _assert_all_finite(token_grads, "token_grads_full")

    answer_grads = token_grads[-T_answer:]             # [T_answer, D]
    peft_model.eval()
    peft_model.zero_grad(set_to_none=True)
    emb.weight.requires_grad_(False)

    raw_scores = torch.linalg.norm(answer_grads.float(), dim=1).cpu().numpy()  # [T_answer]
    print(f"[DEBUG] answer grad-norms: mean={raw_scores.mean():.6f}, max={raw_scores.max():.6f}")
    s = float(np.nansum(raw_scores))
    if not np.isfinite(s) or s == 0.0:
        print("[DEBUG] raw_scores sum invalid or zero:", s, ". Using fallback zeros.")
        normalized_scores = np.zeros_like(raw_scores, dtype=np.float32)
    else:
        relative_significance = raw_scores / s
        normalized_scores = relative_significance * float(total_kl_div.item())

    print(f"[DEBUG] token_scores: mean={normalized_scores.mean():.6f} max={normalized_scores.max():.6f}")
    return normalized_scores.tolist(), kl_full.cpu().numpy().tolist()


# ============================
# Data extraction & caching
# ============================

def load_dataset(filepath: str) -> pd.DataFrame:
    print(f"Loading dataset from {filepath}…")
    with open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"[DEBUG] dataset rows={len(df)}, columns={list(df.columns)}")
    print("[DEBUG] label distribution:\n", df["label"].value_counts())
    return df


def extract_and_cache_probe_data(
    peft_model: PeftModel,
    tokenizer,
    df: pd.DataFrame,
    lora_module_names: List[str],
    cache_path: str,
) -> List[ProbeItem]:
    if os.path.exists(cache_path):
        print(f"Loading cached probe data from {cache_path}…")
        items = torch.load(cache_path, weights_only=False)
        print(f"Loaded cache: {cache_path} | items={len(items)}")
        return items

    print("Extracting probe data (LoRA scalars, token scores, and KL)…")

    if MAX_PROMPTS_TO_PROCESS is not None:
        print(f"[DEBUG] Shuffling and limiting to {MAX_PROMPTS_TO_PROCESS} prompts.")
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True).head(MAX_PROMPTS_TO_PROCESS)

    device = next(peft_model.parameters()).device
    all_items: List[ProbeItem] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt_text = row["prompt"]
        label = row["label"]

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids  # [1, T_prompt]

        # Deterministic generation for stability
        with torch.no_grad():
            generated_ids = peft_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids = generated_ids[:, input_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Compute token significance & KL
        answer_token_scores, prompt_kl_divs = compute_token_scores_and_kl(
            peft_model=peft_model,
            input_ids=input_ids,
            response_ids=response_ids,
        )

        # Trigger hooks over the full seq to capture per-module scalars
        global captured_scalars
        captured_scalars.clear()
        with torch.no_grad():
            peft_model(input_ids=generated_ids)

        if not captured_scalars:
            raise RuntimeError("No LoRA scalars captured — check hooks and model inputs.")

        any_tensor = next(iter(captured_scalars.values()))
        seq_len = any_tensor.shape[1]
        layer_sums: Dict[int, torch.Tensor] = {ln: torch.zeros(seq_len) for ln in LAYER_NUMBERS}
        layer_counts: Dict[int, int] = {ln: 0 for ln in LAYER_NUMBERS}

        for name in lora_module_names:
            scal = captured_scalars.get(name)
            if scal is None:
                continue
            layer_num = get_layer_number_from_name(name)
            if layer_num < 0:
                continue
            layer_sums[layer_num] += scal[0].cpu()
            layer_counts[layer_num] += 1

        prompt_scalars_list: List[LayerScalars] = []
        for t in range(seq_len):
            d: Dict[str, float] = {}
            for ln in LAYER_NUMBERS:
                cnt = max(1, layer_counts[ln])
                d[str(ln)] = float((layer_sums[ln][t] / cnt).item())
            prompt_scalars_list.append(LayerScalars(layer_scalars=d))

        item = ProbeItem(
            prompt=prompt_text,
            label=label,
            response=response_text,
            prompt_answer_first_token_idx=int(input_ids.shape[-1]),
            lora_scalars=PromptScalars(scalars=prompt_scalars_list),
            answer_token_scores=answer_token_scores,
            prompt_kl_divs=prompt_kl_divs,
        )
        all_items.append(item)

    torch.save(all_items, cache_path)
    print(f"Probe data cached at {cache_path} | items={len(all_items)}")
    return all_items


# ============================
# Plots (now saved to disk)
# ============================

def plot_kl_histogram(items: List[ProbeItem], bins: int = 200, save_dir: str = SAVE_DIR, run_id: str = RUN_ID) -> None:
    vals: List[float] = []
    for it in items:
        vals.extend(it.prompt_kl_divs)
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        print("Warning: All KL divergence values are non-finite!")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(vals, bins=bins)
    ax.set_yscale("log")
    ax.set_title("Distribution of KL Divergences (prompt + answer tokens)")
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Count (log scale)")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    path = os.path.join(save_dir, f"kl_hist_{run_id}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[DEBUG] KL histogram saved -> {path}")
    print("KL stats — mean: {:.6f}, median: {:.6f}, std: {:.6f}, min: {:.6f}, max: {:.6f}".format(
        np.mean(vals), np.median(vals), np.std(vals), np.min(vals), np.max(vals)
    ))


def plot_log_token_score_histogram(items: List[ProbeItem], bins: int = 200, save_dir: str = SAVE_DIR, run_id: str = RUN_ID) -> None:
    scores: List[float] = []
    for it in items:
        scores.extend(it.answer_token_scores)
    if len(scores) == 0:
        print("[DEBUG] No answer token scores to plot.")
        return
    log_scores = np.log(np.asarray(scores) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(log_scores, bins=bins)
    ax.set_title("Distribution of Log Answer Token Scores (answer tokens only)")
    ax.set_xlabel("Log Answer Token Score")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    path = os.path.join(save_dir, f"log_token_scores_hist_{run_id}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[DEBUG] Log token score histogram saved -> {path}")
    print(f"[DEBUG] token score log-stats: mean={log_scores.mean():.6f}, max={log_scores.max():.6f}")


def get_regression_coefficient_plot(
    items: List[ProbeItem],
    l1_reg_c: float = 0.01,
    fraction_of_data_to_use: float = 1.0,
    token_kl_div_threshold: Optional[float] = None,
    log_token_score_threshold: Optional[float] = None,
    include_token_features: bool = False,
    n_runs: int = 20,
    z_score_normalize: bool = True,
    figsize: Tuple[int, int] = (14, 6),
    save_dir: str = SAVE_DIR,
    run_id: str = RUN_ID,
    csv_name: Optional[str] = None,
):
    all_coefs: List[np.ndarray] = []

    for _ in tqdm(range(n_runs), desc="Collecting coefficients"):
        X, y, n0, n1 = get_probe_data(
            items,
            balanced_classes=True,
            fraction_of_data_to_use=fraction_of_data_to_use,
            token_kl_div_threshold=token_kl_div_threshold,
            log_token_score_threshold=log_token_score_threshold,
            include_token_features=include_token_features,
            z_score_normalize=z_score_normalize,
        )
        print(f"[DEBUG] get_probe_data -> X={X.shape} y_pos={(y==1).sum()} y_neg={(y==0).sum()}")
        if len(y) < 4:
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        model = LogisticRegression(
            random_state=SEED,
            max_iter=1000,
            penalty="l1",
            solver="liblinear",
            C=l1_reg_c,
        )
        model.fit(X_train, y_train)
        all_coefs.append(model.coef_[0])

    if not all_coefs:
        raise RuntimeError("No coefficients collected — check filtering thresholds.")

    coef_mat = np.vstack(all_coefs)

    # Column names: one per layer (+ optional token score feature)
    col_names = [f"L{ln}" for ln in LAYER_NUMBERS]
    if include_token_features:
        col_names = col_names + ["token_score"]

    # ---- SAVE COEFFICIENTS CSV ----
    df = pd.DataFrame(coef_mat, columns=col_names)
    csv_path = os.path.join(save_dir, csv_name or f"coefficients_{run_id}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DEBUG] Coefficients CSV saved -> {csv_path}")

    # Diagnostics: sparsity and mean±sd
    nz_ratio = (coef_mat != 0).mean(axis=0)
    mean_coef = coef_mat.mean(axis=0)
    std_coef  = coef_mat.std(axis=0)
    print("[DEBUG] Non-zero % by feature:",
          {n: round(float(r*100), 1) for n, r in zip(col_names, nz_ratio)})
    print("[DEBUG] Mean±SD coeff by feature:",
          {n: f"{m:.4f}±{s:.4f}" for n, m, s in zip(col_names, mean_coef, std_coef)})

    # ---- SAVE VIOLIN PLOT ----
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    sns.violinplot(data=df, ax=ax, inner="quartile", linewidth=0.8, width=0.8)
    ax.set_title("Distribution of Logistic Regression Coefficients (binary task)")
    ax.set_xlabel("Layer / Feature")
    ax.set_ylabel("Coefficient value")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, f"coef_violin_{run_id}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[DEBUG] Coefficient violin saved -> {fig_path}")


# ============================
# Feature construction
# ============================

def get_single_class_probe_data(
    items: List[ProbeItem],
    class_label_int: int,
    token_kl_div_threshold: Optional[float] = None,
    log_token_score_threshold: Optional[float] = None,
    include_token_features: bool = False,
) -> List[List[float]]:
    feats: List[List[float]] = []

    for it in items:
        y = 1 if it.label == LABEL_POSITIVE else 0
        if y != class_label_int:
            continue

        first_idx = it.prompt_answer_first_token_idx
        for i, token_score in enumerate(it.answer_token_scores):
            token_idx = first_idx + i

            if token_kl_div_threshold is not None and it.prompt_kl_divs[token_idx] < token_kl_div_threshold:
                continue
            if log_token_score_threshold is not None and np.log(token_score + 1e-9) < log_token_score_threshold:
                continue

            layer_dict = it.lora_scalars.scalars[token_idx].layer_scalars
            vec = [layer_dict[str(ln)] for ln in LAYER_NUMBERS]
            if include_token_features:
                vec.append(float(token_score))
            feats.append(vec)

    return feats


def get_probe_data(
    items: List[ProbeItem],
    balanced_classes: bool = True,
    fraction_of_data_to_use: float = 1.0,
    token_kl_div_threshold: Optional[float] = None,
    log_token_score_threshold: Optional[float] = None,
    include_token_features: bool = False,
    z_score_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, int]:

    class0_feats = get_single_class_probe_data(
        items, 0, token_kl_div_threshold, log_token_score_threshold, include_token_features
    )
    class1_feats = get_single_class_probe_data(
        items, 1, token_kl_div_threshold, log_token_score_threshold, include_token_features
    )

    X0 = np.asarray(class0_feats, dtype=np.float32)
    X1 = np.asarray(class1_feats, dtype=np.float32)

    if fraction_of_data_to_use < 1.0:
        rng = np.random.default_rng(SEED)
        n0 = int(len(X0) * fraction_of_data_to_use)
        n1 = int(len(X1) * fraction_of_data_to_use)
        X0 = X0[rng.permutation(len(X0))[:n0]]
        X1 = X1[rng.permutation(len(X1))[:n1]]

    if balanced_classes and len(X0) and len(X1):
        n = min(len(X0), len(X1))
        rng = np.random.default_rng()
        X0 = X0[rng.permutation(len(X0))[:n]]
        X1 = X1[rng.permutation(len(X1))[:n]]

    if len(X0) and len(X1):
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))]).astype(np.float32).ravel()
    else:
        X = X0 if len(X1) == 0 else X1
        y = (np.zeros(len(X)) if len(X1) == 0 else np.ones(len(X))).astype(np.float32).ravel()

    if len(y):
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]

    if z_score_normalize and len(y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X)

    size0 = int((y == 0).sum())
    size1 = int((y == 1).sum())
    return X, y, size0, size1


# ============================
# Metrics & percentage utility
# ============================

def get_regression_metrics(
    items: List[ProbeItem],
    l1_reg_c: float = 0.01,
    fraction_of_data_to_use: float = 1.0,
    token_kl_div_threshold: Optional[float] = None,
    log_token_score_threshold: Optional[float] = None,
    include_token_features: bool = False,
    n_runs: int = 20,
    z_score_normalize: bool = True,
) -> AveragedRegressionMetrics:
    accuracies: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    confs: List[np.ndarray] = []
    reports: List[str] = []

    for _ in tqdm(range(n_runs), desc="Running logistic regression"):
        X, y, _, _ = get_probe_data(
            items,
            balanced_classes=True,
            fraction_of_data_to_use=fraction_of_data_to_use,
            token_kl_div_threshold=token_kl_div_threshold,
            log_token_score_threshold=log_token_score_threshold,
            include_token_features=include_token_features,
            z_score_normalize=z_score_normalize,
        )
        print(f"[DEBUG] get_probe_data -> X={X.shape} y_pos={(y==1).sum()} y_neg={(y==0).sum()}")
        if len(y) < 4:
            raise RuntimeError("Not enough samples after filtering to run regression.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )

        model = LogisticRegression(
            random_state=SEED,
            max_iter=1000,
            penalty="l1",
            solver="liblinear",
            C=l1_reg_c,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracies.append(float(accuracy_score(y_test, y_pred)))
        precisions.append(float(precision_score(y_test, y_pred)))
        recalls.append(float(recall_score(y_test, y_pred)))
        f1s.append(float(f1_score(y_test, y_pred)))
        aucs.append(float(roc_auc_score(y_test, y_proba)))
        confs.append(confusion_matrix(y_test, y_pred))
        reports.append(classification_report(y_test, y_pred, target_names=[LABEL_NEGATIVE, LABEL_POSITIVE]))

    return AveragedRegressionMetrics(
        accuracy=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        precision=float(np.mean(precisions)),
        precision_std=float(np.std(precisions)),
        recall=float(np.mean(recalls)),
        recall_std=float(np.std(recalls)),
        f1_score=float(np.mean(f1s)),
        f1_score_std=float(np.std(f1s)),
        auc_roc=float(np.mean(aucs)),
        auc_roc_std=float(np.std(aucs)),
        confusion_matrix=np.mean(confs, axis=0),
        classification_report=reports[-1],
    )


def percent_tokens_above_log_score_threshold(items: List[ProbeItem], threshold: float = 0.5) -> float:
    total = 0
    above = 0
    for it in items:
        logs = np.log(np.asarray(it.answer_token_scores) + 1e-9)
        above += int((logs > threshold).sum())
        total += logs.size
    pct = (above / max(1, total)) * 100.0
    print(f"{pct:.2f}% of answer tokens have log(token score) > {threshold}")
    return pct


# ============================
# Main
# ============================

def main():
    setup_environment()

    # Load single model + adapter (forces eager attention + sanity check)
    peft_model, tokenizer = load_model_and_tokenizer()

    # Attach hooks and determine layer ordering
    lora_module_names, handles = add_hooks_to_model(peft_model)

    # Load data & build/cache probe items
    df = load_dataset(DATASET_PATH)
    probe_items = extract_and_cache_probe_data(
        peft_model=peft_model,
        tokenizer=tokenizer,
        df=df,
        lora_module_names=lora_module_names,
        cache_path=CACHE_PATH,
    )

    # Clean up hooks
    remove_hooks(handles)

    # --- Plots (saved to SAVE_DIR) ---
    plot_kl_histogram(probe_items, bins=200, save_dir=SAVE_DIR, run_id=RUN_ID)
    plot_log_token_score_histogram(probe_items, bins=200, save_dir=SAVE_DIR, run_id=RUN_ID)

    # --- Metrics ---
    metrics = get_regression_metrics(
        items=probe_items,
        l1_reg_c=0.01,
        fraction_of_data_to_use=1.0,
        token_kl_div_threshold=None,
        log_token_score_threshold=1.2,
        include_token_features=False,
        n_runs=NUM_REGRESSION_RUNS,
        z_score_normalize=True,
    )

    print("\n=== Averaged Regression Metrics ===")
    print(f"Accuracy:  {metrics.accuracy:.3f} ± {metrics.accuracy_std:.3f}")
    print(f"Precision: {metrics.precision:.3f} ± {metrics.precision_std:.3f}")
    print(f"Recall:    {metrics.recall:.3f} ± {metrics.recall_std:.3f}")
    print(f"F1-score:  {metrics.f1_score:.3f} ± {metrics.f1_score_std:.3f}")
    print(f"AUC-ROC:   {metrics.auc_roc:.3f} ± {metrics.auc_roc_std:.3f}")
    print("\nAveraged Confusion Matrix (mean over runs):\n", metrics.confusion_matrix)
    print("\nLast Classification Report:\n", metrics.classification_report)

    # --- Coefficient violin + CSV dump (saved to SAVE_DIR) ---
    get_regression_coefficient_plot(
        items=probe_items,
        l1_reg_c=0.01,
        fraction_of_data_to_use=1.0,
        token_kl_div_threshold=None,
        log_token_score_threshold=1.2,
        include_token_features=False,
        n_runs=100,
        z_score_normalize=True,
        figsize=(12, 5),
        save_dir=SAVE_DIR,
        run_id=RUN_ID,
        csv_name=None,  # default "coefficients_<RUN_ID>.csv"
    )

    # --- Percentage utility (prints only) ---
    percent_tokens_above_log_score_threshold(probe_items, threshold=0.0)
    percent_tokens_above_log_score_threshold(probe_items, threshold=-1.0)


if __name__ == "__main__":
    main()
