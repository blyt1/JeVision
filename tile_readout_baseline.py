"""Zero-shot tile defect readout (the "visual Jev" baseline).

For each tile: one forward pass through a frozen vision-language model, no text
generation. The score is read from the logits of "Yes" vs "No" at the position
where the answer would start.

Usage:
    pip install "transformers>=4.57" accelerate torch pillow scikit-learn
    python tile_readout_baseline.py --jsonl test_250k.jsonl --image-root /data/tiles

Adjust --image-key / --label-key to match your JSONL fields. The label field is
expected to look like "defect: <name>; description: <text>", with "ok" meaning
no defect.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTIONS = {
    "v1": (
        "This is a close-up tile cropped from a photo of a manufactured part. "
        "Is there a visible defect in this tile? Answer Yes or No."
    ),
    "v2": (
        "This is a small tile cropped from a larger photo of a manufactured part "
        "or surface. Most tiles are normal. A defective tile shows damage or an "
        "anomaly such as a crack, scratch, dent, hole, stain, oxidation, a "
        "hotspot, a broken or missing piece, or an overlap. Textures, edges and "
        "background are normal. Does this tile show a defect? Answer Yes or No."
    ),
    "v3": (
        "You are a quality-inspection system. Look closely at this tile and "
        "compare it with how an undamaged part would look. Is anything abnormal, "
        "damaged, discoloured or missing? Answer Yes or No."
    ),
}


def load_rows(path, image_key, label_key, ok_label):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            defect = r[label_key].split(";")[0].replace("defect:", "").strip().lower()
            rows.append((r[image_key], defect, int(defect != ok_label)))
    return rows


def single_token_ids(tokenizer, words):
    """Token ids for the answer words that encode to exactly one token."""
    ids = []
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.append(t[0])
    if not ids:
        raise ValueError(f"None of {words} is a single token for this tokenizer")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--label-key", default="suffix")
    ap.add_argument("--ok-label", default="ok")
    ap.add_argument("--n", type=int, default=2000, help="tiles to sample (0 = all)")
    ap.add_argument("--tile-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--questions", default="v1,v2,v3", help="comma-separated keys of QUESTIONS")
    ap.add_argument("--out", default="tile_scores.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.jsonl, args.image_key, args.label_key, args.ok_label)
    random.Random(args.seed).shuffle(rows)
    if args.n:
        rows = rows[: args.n]
    print(f"{len(rows)} tiles, {sum(r[2] for r in rows)} labelled defective")

    processor = AutoProcessor.from_pretrained(args.model)
    # Left padding so the last position of every row is the answer slot.
    processor.tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16  # Apple Silicon
    else:
        device, dtype = "cpu", torch.float32
    print(f"Running on {device} ({dtype})")
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    yes_ids = single_token_ids(processor.tokenizer, ["Yes", "yes"])
    no_ids = single_token_ids(processor.tokenizer, ["No", "no"])

    names = args.questions.split(",")
    margins = {q: [] for q in names}
    root = Path(args.image_root)
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        imgs = [
            Image.open(root / name).convert("RGB").resize((args.tile_size,) * 2)
            for name, _, _ in batch
        ]
        for q in names:
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": QUESTIONS[q]},
                        ],
                    }
                ]
                for img in imgs
            ]
            inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            with torch.inference_mode():
                last = model(**inputs).logits[:, -1, :].float()
            yes = torch.logsumexp(last[:, yes_ids], dim=1)
            no = torch.logsumexp(last[:, no_ids], dim=1)
            margins[q].extend((yes - no).cpu().tolist())
        print(f"\r{min(i + args.batch_size, len(rows))}/{len(rows)}", end="")
    print()

    labels = [r[2] for r in rows]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "defect", "label"] + [f"margin_{q}" for q in names])
        for j, (name, defect, y) in enumerate(rows):
            w.writerow([name, defect, y] + [margins[q][j] for q in names])

    # Ranking metrics only: these margins are not calibrated probabilities.
    for q in names:
        fpr, tpr, _ = roc_curve(labels, margins[q])
        print(f"[{q}] AUROC {roc_auc_score(labels, margins[q]):.3f}  "
              f"AP {average_precision_score(labels, margins[q]):.3f}  "
              f"recall@{args.target_fpr:.0%}FPR {tpr[fpr <= args.target_fpr].max():.3f}")
    print(f"Scores written to {args.out}")


if __name__ == "__main__":
    main()
