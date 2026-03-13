import argparse, json, random, math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Sequence
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, logging as hf_log

hf_log.set_verbosity_error()  # keep stdout clean

from collections import defaultdict

from chronos import BaseChronosPipeline, ChronosPipeline

import torch.nn as nn

import numpy as np

# Data utilities
def iter_series(dataset_name, num_series, series_len, seed=0):
    ds = load_dataset(
        "autogluon/chronos_datasets",
        dataset_name,
        split="train",
        keep_in_memory=False,
    )
    ds.set_format("numpy")

    # identify numeric sequence columns once
    seq_cols = [
        col for col, feat in ds.features.items()
        if isinstance(feat, Sequence) and feat.feature.dtype in ("float32", "float64", "int64", "int32")
    ]
    if not seq_cols:
        raise ValueError(f"No numeric sequence column found in {dataset_name}")
    col = seq_cols[0]

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(num_series, len(ds)))
    for idx in indices:
        seq = ds[idx][col].astype(np.float32)
        if len(seq) < series_len:
            continue
        yield seq[-series_len:]


# Jaccard‑overlap metric
def shift_right(ids, model_base):
    return torch.cat(
        [torch.full_like(ids[:, :1], model_base.config.decoder_start_token_id), ids[:, :-1]],
        dim=1,
    )

@torch.no_grad()
def topk_jaccard_chronos(model_base, model_cmp, tokenizer, series_list,
                         topk=10, device="cuda"):
    model_base.to(device).eval()
    model_cmp.to(device).eval()
    overlaps, count = 0.0, 0
    for ts in tqdm(series_list):
        ts   = torch.from_numpy(ts).unsqueeze(0)          # [1, L]
        ids, mask, _ = tokenizer.context_input_transform(ts)
        ids, mask = ids.to(device), mask.to(device)
        dec_in = shift_right(ids, model_base)

        logit_b = model_base(input_ids=ids, attention_mask=mask,
                       decoder_input_ids=dec_in).logits.squeeze(0)
        logit_c = model_cmp(input_ids=ids, attention_mask=mask,
                       decoder_input_ids=dec_in).logits.squeeze(0)

        pad = ids.squeeze(0).eq(tokenizer.config.pad_token_id)
        top_b = logit_b.topk(topk, dim=-1).indices
        top_c = logit_c.topk(topk, dim=-1).indices
        for t in range(ids.size(1)):
            if pad[t]: continue
            s1, s2 = set(top_b[t].tolist()), set(top_c[t].tolist())
            overlaps += len(s1 & s2) / len(s1 | s2)
            count   += 1
    return overlaps / count

# Compression methods
def compression(epsilon):
    model = ChronosPipeline.from_pretrained("amazon/chronos-t5-base").model.model
    ranks = apply_svd_to_t5_attention(model, epsilon)

    return model, ranks

def low_rank_approximation(W: torch.Tensor, epsilon: float):
    # Compute SVD
    U, S, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    # Relative thresholding
    S_rel = S / S[0]
    rank = (S_rel >= epsilon).sum().item()
    # Truncate
    U_trunc = U[:, :rank]
    S_trunc = S[:rank]
    Vh_trunc = Vh[:rank, :]
    # Reconstruct
    W_approx = (U_trunc * S_trunc) @ Vh_trunc
    return W_approx.to(torch.bfloat16), rank

def apply_svd_to_t5_attention(model, epsilon=1e-3):
    ranks = defaultdict(list)  # Keep track of ranks per projection type

    for name, module in model.named_modules():
        if hasattr(module, 'q') and isinstance(module.q, nn.Linear):
            for proj_name in ['q', 'k', 'v', 'o']:
                W = getattr(module, proj_name).weight.data
                W_approx, rank = low_rank_approximation(W, epsilon)
                getattr(module, proj_name).weight.data.copy_(W_approx)
                ranks[proj_name].append((name, rank))
    
    return ranks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       type=str, required=True,
                   help="e.g. electricity_15min, m4_daily, taxi_30min, uber_tlc_hourly")
    p.add_argument("--num-series",    type=int, default=1000)
    p.add_argument("--series-len",    type=int, default=512)
    p.add_argument("--topk",          type=int, default=10)
    p.add_argument("--device",        type=str, default="cuda")
    args = p.parse_args()


    pipe_base = ChronosPipeline.from_pretrained("amazon/chronos-t5-base")
    tokenizer = pipe_base.tokenizer
    base_model = pipe_base.model.model

    epsilons = [0, 1e-3, 10 ** -2.5, 10 ** -2, 10 ** -1.5, 10 ** -1, 10 ** -0.75, 10 ** -0.5]

    counter = 0
    for epsilon in epsilons:
        cmp_model, rank  = compression(epsilon)
        cmp_model = cmp_model

        print("Ranks are reduced to " + str(rank) + "\n")

        series_iter = iter_series(
            args.dataset, args.num_series, args.series_len, seed=0
        )

        score = topk_jaccard_chronos(
            base_model, cmp_model, tokenizer,
            series_iter, topk=args.topk, device=args.device
        )

        outfile = f"./results/chronos_{counter}.json"

        result = {
            "chronos_model": "amazon/chronos-t5-base",
            "dataset": args.dataset,
            "num_series": args.num_series,
            "series_len": args.series_len,
            "compression": ratio,
            f"top{args.topk}_jaccard": score,
        }

        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2)

        print(json.dumps(result, indent=2))
        print(f"Saved -> {outfile}")
        counter += 1


if __name__ == "__main__":
    main()