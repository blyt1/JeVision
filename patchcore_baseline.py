"""PatchCore baseline on the same tiles as tile_scores.csv.

Per-product memory banks (product = filename prefix) built from 'ok' tiles in the
train split; tile score = max over patches of the distance to the nearest bank patch.
Features: WideResNet50 layer2+layer3, 3x3 local average pooling (as in PatchCore).
"""
import argparse, collections, csv, hashlib, json, random, re
from pathlib import Path

import torch, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

ap = argparse.ArgumentParser()
ap.add_argument("--train-jsonl", default="/Users/benjamintang/Downloads/train_250k.jsonl")
ap.add_argument("--scores", default="tile_scores.csv", help="tiles to score (same sample as the VLM run)")
ap.add_argument("--image-root", default="all_data")
ap.add_argument("--per-product", type=int, default=100, help="ok train tiles per product bank")
ap.add_argument("--min-train", type=int, default=10)
ap.add_argument("--coreset", type=float, default=0.1)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--out", default="patchcore_scores.csv")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
prod = lambda n: re.split(r"[_\d(. ]", n)[0]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
root = Path(args.image_root)

test = list(csv.DictReader(open(args.scores)))
test_hashes = {md5(root / r["image"]) for r in test}

bank_files = collections.defaultdict(list)
for l in open(args.train_jsonl):
    r = json.loads(l)
    if r["suffix"].split(";")[0].replace("defect:", "").strip().lower() == "ok":
        bank_files[prod(r["image"])].append(r["image"])
rng = random.Random(0)

net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.inference_mode()
def feats(names):
    x = torch.stack([
        torch.from_numpy(__import__("numpy").asarray(
            Image.open(root / n).convert("RGB").resize((args.size,) * 2))).permute(2, 0, 1)
        for n in names]).float().div(255).to(dev)
    x = (x - mean) / std
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    f2 = net.layer2(net.layer1(x)); f3 = net.layer3(f2)
    f2, f3 = (F.avg_pool2d(f, 3, 1, 1) for f in (f2, f3))
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([f2, f3], 1)                      # B, 1536, H, W
    return f.flatten(2).transpose(1, 2)             # B, H*W, 1536


def coreset(x, frac, proj=128):
    """Greedy k-center on randomly projected patches."""
    n = max(1, int(len(x) * frac))
    z = x @ torch.randn(x.shape[1], proj, device=x.device) / proj ** 0.5
    sel = [random.randrange(len(z))]
    d = torch.cdist(z, z[sel[-1:]]).squeeze(1)
    for _ in range(n - 1):
        sel.append(int(d.argmax()))
        d = torch.minimum(d, torch.cdist(z, z[sel[-1:]]).squeeze(1))
    return x[sel]


banks = {}
for p in {prod(r["image"]) for r in test}:
    files = bank_files.get(p, [])
    rng.shuffle(files)
    files = [f for f in files[: args.per_product * 2] if md5(root / f) not in test_hashes][: args.per_product]
    if len(files) < args.min_train:
        continue
    allf = torch.cat([feats(files[i:i + 16]).flatten(0, 1) for i in range(0, len(files), 16)])
    banks[p] = coreset(allf, args.coreset)
    print(f"bank {p}: {len(files)} tiles -> {len(banks[p])} patches", flush=True)

rows, labels, scores = [], [], []
for i, r in enumerate(test):
    p = prod(r["image"])
    if p not in banks:
        continue
    f = feats([r["image"]])[0]
    s = torch.cdist(f, banks[p]).min(1).values.max().item()
    rows.append(r); labels.append(int(r["label"])); scores.append(s)
    if i % 100 == 0:
        print(f"\r{i}/{len(test)}", end="", flush=True)
print()
with open(args.out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["image", "defect", "label", "patchcore"])
    for r, s in zip(rows, scores):
        w.writerow([r["image"], r["defect"], r["label"], s])
fpr, tpr, _ = roc_curve(labels, scores)
print(f"scored {len(rows)}/{len(test)} tiles (products with no ok train bank skipped), {sum(labels)} defective")
print(f"PatchCore AUROC {roc_auc_score(labels, scores):.3f}  AP {average_precision_score(labels, scores):.3f}  "
      f"recall@5%FPR {tpr[fpr <= 0.05].max():.3f}")
