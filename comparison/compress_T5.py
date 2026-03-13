import argparse, json, math, os, random
from copy import deepcopy
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    logging as hf_logging,
)

from collections import defaultdict

import torch.nn as nn

import numpy as np

hf_logging.set_verbosity_error()


# helpers
@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, batch_size=8, device="cuda"):
    model.eval().to(device)
    total_nll, n_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc="PPL"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)

        labels = enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        loss = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        ).loss

        n_tok  = (labels != -100).sum().item()
        total_nll += loss.item() * n_tok
        n_tokens  += n_tok

    return math.exp(total_nll / n_tokens)


@torch.no_grad()
def topk_jaccard_overlap(
    model1, model2, tokenizer, texts, topk=10, batch_size=8, device="cuda"
):
    """Compute top-k Jaccard overlap between two models"""
    model1.eval().to(device)
    model2.eval().to(device)

    overlaps, counts = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Jaccard@{topk}"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Build decoder_input_ids exactly the way the loss function would.
        decoder_input_ids = model1._shift_right(enc["input_ids"])

        logits1 = model1(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        ).logits

        logits2 = model2(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        ).logits

        pad_mask = enc["input_ids"].eq(tokenizer.pad_token_id)
        top1 = logits1.topk(topk, dim=-1).indices
        top2 = logits2.topk(topk, dim=-1).indices
        for b in range(top1.size(0)):
            for t in range(top1.size(1)):
                if pad_mask[b, t]:
                    continue
                s1, s2 = set(top1[b, t].tolist()), set(top2[b, t].tolist())
                if s1 or s2:
                    overlaps += len(s1 & s2) / len(s1 | s2)
                    counts += 1
    return overlaps / max(counts, 1)


def load_c4_validation(num_sentences=1000, seed=0):
    ds = load_dataset("c4", "en", split="validation", streaming=True)
    random.seed(seed)
    texts = []
    for ex in ds:
        text = ex["text"].strip()
        if len(text) > 0:
            texts.append(text)
        if len(texts) >= num_sentences:
            break
    return texts


def compression(ratio):
    model = T5ForConditionalGeneration.from_pretrained("google/t5-base-lm-adapt")
    ranks = apply_svd_to_t5_attention(model, ratio)

    return model, ranks

def low_rank_approximation(W: torch.Tensor, ratio: float):
    # Compute SVD
    U, S, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    # Relative thresholding
    rank = int(S.shape[0] * ratio)
    # Truncate
    U_trunc = U[:, :rank]
    S_trunc = S[:rank]
    Vh_trunc = Vh[:rank, :]
    # Reconstruct
    W_approx = (U_trunc * S_trunc) @ Vh_trunc
    return W_approx.to(torch.bfloat16), rank


def apply_svd_to_t5_attention(model, ratio):
    ranks = defaultdict(list)  # Keep track of ranks per projection type

    for name, module in model.named_modules():
        if hasattr(module, 'q') and isinstance(module.q, nn.Linear):
            for proj_name in ['q', 'k', 'v', 'o']:
                W = getattr(module, proj_name).weight.data
                W_approx, rank = low_rank_approximation(W, ratio)
                getattr(module, proj_name).weight.data.copy_(W_approx)
                ranks[proj_name].append((name, rank))
    
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-sentences", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    # 1. load baseline
    tokenizer = AutoTokenizer.from_pretrained("google/t5-base-lm-adapt")
    baseline = T5ForConditionalGeneration.from_pretrained("google/t5-base-lm-adapt")

    ratios = [1, 0.889, 0.755, 0.569, 0.393, 0.237, 0.151, 0.073]

    counter = 0

    for ratio in ratios:

        print("Reduction ratio is " + str(ratio))

        # compress
        compressed, rank = compression(ratio)

        print("Ranks are reduced to " + str(rank) + "\n")

        # data
        texts = load_c4_validation(num_sentences=args.num_sentences)

        # metrics
        ppl = compute_perplexity(compressed, tokenizer, texts,
                                batch_size=args.batch_size, device=args.device)

        print(f"perplexity = {ppl}")
        
        jacc = topk_jaccard_overlap(
            baseline,
            compressed,
            tokenizer,
            texts,
            topk=args.topk,
            batch_size=args.batch_size,
            device=args.device,
        )

        result = {
            "model": "t5-base",
            "compression": ratio,
            "num_sentences": args.num_sentences,
            "perplexity": ppl,
            f"top{args.topk}_jaccard": jacc,
        }
        outfile = f"./results/T5_{counter}.json"
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nSaved -> {outfile}")
        print(json.dumps(result, indent=2))

        counter += 1


if __name__ == "__main__":
    main()