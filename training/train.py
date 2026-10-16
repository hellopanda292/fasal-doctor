# %% [markdown]
# # Fasal Doctor: crop disease detection with transfer learning
#
# Recognises **17 classes** of leaves (healthy or diseased) for three crops grown in
# Uttar Pradesh: **potato (आलू), tomato (टमाटर) and maize (मक्का)**.
#
# **Approach: transfer learning.** A MobileNetV3 network pretrained on ImageNet already knows
# how to see edges, textures and spots. We fine-tune it on PlantVillage leaf photos:
# 1. train only a new classifier head on frozen features (fast, on a laptop CPU),
# 2. then fine-tune the last blocks of the network for a few epochs.
#
# **Honest evaluation.** PlantVillage contains several photos of the *same physical leaf*.
# A random split would put photos of one leaf in both train and test and inflate accuracy.
# We split **by leaf** (using the leaf ID in each filename), so every test leaf is unseen.
#
# Data: PlantVillage (Hughes & Salathé, 2015), CC BY-SA 3.0. Models trained on it are shared
# under the same licence.

# %%
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))

ROOT = Path.cwd().parent if Path.cwd().name == "training" else Path.cwd()
IMAGES = ROOT / "data" / "pv" / "raw" / "color"
DOCS = ROOT / "docs"
WEB_MODEL = ROOT / "web" / "model"
DOCS.mkdir(exist_ok=True)
WEB_MODEL.mkdir(parents=True, exist_ok=True)
IMG = 224

# %% [markdown]
# ## 1. Classes, with Hindi names and simple advice for farmers

# %%
CLASSES = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("मक्का", "ग्रे लीफ स्पॉट", "Maize", "Gray leaf spot"),
    "Corn_(maize)___Common_rust_": ("मक्का", "रतुआ (कॉमन रस्ट)", "Maize", "Common rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("मक्का", "उत्तरी पत्ती झुलसा", "Maize", "Northern leaf blight"),
    "Corn_(maize)___healthy": ("मक्का", "स्वस्थ", "Maize", "Healthy"),
    "Potato___Early_blight": ("आलू", "अगेती झुलसा", "Potato", "Early blight"),
    "Potato___Late_blight": ("आलू", "पछेती झुलसा", "Potato", "Late blight"),
    "Potato___healthy": ("आलू", "स्वस्थ", "Potato", "Healthy"),
    "Tomato___Bacterial_spot": ("टमाटर", "बैक्टीरियल धब्बा", "Tomato", "Bacterial spot"),
    "Tomato___Early_blight": ("टमाटर", "अगेती झुलसा", "Tomato", "Early blight"),
    "Tomato___Late_blight": ("टमाटर", "पछेती झुलसा", "Tomato", "Late blight"),
    "Tomato___Leaf_Mold": ("टमाटर", "पत्ती फफूंद", "Tomato", "Leaf mould"),
    "Tomato___Septoria_leaf_spot": ("टमाटर", "सेप्टोरिया पत्ती धब्बा", "Tomato", "Septoria leaf spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("टमाटर", "लाल मकड़ी (माइट)", "Tomato", "Spider mites"),
    "Tomato___Target_Spot": ("टमाटर", "टारगेट स्पॉट", "Tomato", "Target spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("टमाटर", "पत्ती मरोड़ (पीला) वायरस", "Tomato", "Yellow leaf curl virus"),
    "Tomato___Tomato_mosaic_virus": ("टमाटर", "मोज़ेक वायरस", "Tomato", "Mosaic virus"),
    "Tomato___healthy": ("टमाटर", "स्वस्थ", "Tomato", "Healthy"),
}
FOLDERS = sorted(CLASSES)
print(len(FOLDERS), "classes")

# %% [markdown]
# ## 2. Files and the leaf-level split
#
# Filenames look like `0a8a…___RS_Erly.B 7557.JPG`: a photo source prefix and a photo number.
# The PlantVillage authors published `leaf-map.json`, which says which physical leaf each photo
# shows (about 4 photos per leaf). Where a photo is in that map we group by the real leaf ID.
# For the rest (all maize, mosaic virus and some tomato sources) we group photos in blocks of 50
# consecutive photo numbers: photos of one leaf are taken one after another, and on the mapped
# photos this keeps about 90% of leaves entirely inside one block.

# %%
def prefix_number(name: str):
    m = re.match(r"^(\D*?)\s*(\d+)", name.strip().lower())
    return (m.group(1).strip(), int(m.group(2))) if m else (name.strip().lower(), 0)


ALIASES = {"com.g_tgs_fl": "com.g_fl"}  # same source, spelled differently in the map
leaf_map = {}
for key, values in json.loads((ROOT / "data" / "leaf-map.json").read_text(encoding="utf-8")).items():
    prefix, number = prefix_number(key)
    for v in values:
        folder, leaf = v.split(":::")
        leaf_map[(folder, prefix, number)] = f"{folder}|leaf {leaf}"


def leaf_id(path: Path):
    prefix, number = prefix_number(path.stem.split("___", 1)[-1])
    prefix = ALIASES.get(prefix, prefix)
    exact = leaf_map.get((path.parent.name, prefix, number))
    return (exact, True) if exact else (f"{path.parent.name}|{prefix}|block {number // 50}", False)


files, labels, leaves, exact = [], [], [], []
for ci, folder in enumerate(FOLDERS):
    for f in sorted((IMAGES / folder).iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lid, ex = leaf_id(f)
            files.append(f); labels.append(ci); leaves.append(lid); exact.append(ex)
labels, exact = np.array(labels), np.array(exact)
print(f"{len(files):,} images in {len(set(leaves)):,} leaf groups; {exact.mean() * 100:.0f}% have an exact leaf ID")

# Split leaves (not images) into train 70% / validation 15% / test 15%, per class.
split = {}
for ci in range(len(FOLDERS)):
    ls = sorted({l for l, y in zip(leaves, labels) if y == ci})
    random.Random(ci).shuffle(ls)
    n = len(ls)
    for i, l in enumerate(ls):
        split[l] = "train" if i < 0.70 * n else "val" if i < 0.85 * n else "test"
part = np.array([split[l] for l in leaves])
for p in ["train", "val", "test"]:
    print(p, int((part == p).sum()))
assert not (set(np.array(leaves)[part == "train"]) & set(np.array(leaves)[part == "test"])), "leaf leakage"

# %%
counts = Counter(labels.tolist())
fig, ax = plt.subplots(figsize=(9, 5))
names = [f"{CLASSES[f][2]}: {CLASSES[f][3]}" for f in FOLDERS]
ax.barh(names, [counts[i] for i in range(len(FOLDERS))], color=["#0f8a45" if "healthy" in f else "#e05a0c" for f in FOLDERS])
ax.set_title("Images per class"); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(DOCS / "classes.png", dpi=110); plt.show()

# %% [markdown]
# ## 3. Data loading and augmentation
#
# Real phone photos are rotated, zoomed and lit differently from lab photos, so training images
# are randomly cropped, flipped, rotated and colour-jittered.

# %%
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Leaves(Dataset):
    def __init__(self, idx, tf):
        self.idx, self.tf = idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(Image.open(files[j]).convert("RGB")), int(labels[j])


idx = {p: np.where(part == p)[0] for p in ["train", "val", "test"]}
loaders = {
    "train": DataLoader(Leaves(idx["train"], train_tf), batch_size=64, shuffle=True, num_workers=0),
    "val": DataLoader(Leaves(idx["val"], eval_tf), batch_size=128, num_workers=0),
    "test": DataLoader(Leaves(idx["test"], eval_tf), batch_size=128, num_workers=0),
}

# %% [markdown]
# ## 4. Model: MobileNetV3-Large, pretrained on ImageNet
#
# Small and fast enough to run inside a phone browser. We replace its last layer with a new
# one for our 17 classes.

# %%
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(FOLDERS))


def run_epoch(loader, train, optimizer=None, max_batches=None):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.05)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += len(y); correct += (out.argmax(1) == y).sum().item(); loss_sum += loss.item() * len(y)
    return loss_sum / total, correct / total


def set_trainable(stage):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if stage == 2:  # also the last blocks of the feature extractor
        for p in model.features[-4:].parameters():
            p.requires_grad = True


history = []
plan = [(1, 2, 1e-3), (2, 3, 2e-4)]  # (stage, epochs, learning rate)
for stage, epochs, lr in plan:
    set_trainable(stage)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    for e in range(epochs):
        t = time.time()
        tl, ta = run_epoch(loaders["train"], True, opt)
        vl, va = run_epoch(loaders["val"], False)
        history.append({"stage": stage, "train_acc": ta, "val_acc": va})
        print(f"stage {stage} epoch {e + 1}: train acc {ta:.4f} | val acc {va:.4f} | {time.time() - t:.0f}s")

torch.save(model.state_dict(), ROOT / "training" / "fasal_doctor.pt")

# %% [markdown]
# ## 5. Test on leaves the model has never seen

# %%
model.eval()
preds, ys = [], []
with torch.no_grad():
    for x, y in loaders["test"]:
        preds.append(model(x).argmax(1)); ys.append(y)
preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
test_acc = (preds == ys).mean()
exact_acc = (preds == ys)[exact[idx["test"]]].mean()
print(f"Test accuracy (unseen leaves): {test_acc * 100:.2f}% on {len(ys):,} images")
print(f"  on test photos with an exact leaf ID: {exact_acc * 100:.2f}%")

per_class = {FOLDERS[c]: float((preds[ys == c] == c).mean()) for c in range(len(FOLDERS))}
for f, a in sorted(per_class.items(), key=lambda kv: kv[1])[:5]:
    print(f"  hardest: {CLASSES[f][2]} {CLASSES[f][3]}: {a * 100:.1f}%")

cm = np.zeros((len(FOLDERS), len(FOLDERS)), int)
for t_, p_ in zip(ys, preds):
    cm[t_, p_] += 1
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(np.log1p(cm), cmap="Greens")
ax.set_xticks(range(len(FOLDERS))); ax.set_yticks(range(len(FOLDERS)))
short = [f"{CLASSES[f][2][:3]} {CLASSES[f][3]}" for f in FOLDERS]
ax.set_xticklabels(short, rotation=90, fontsize=7); ax.set_yticklabels(short, fontsize=7)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix · test accuracy {test_acc * 100:.1f}%")
plt.tight_layout(); plt.savefig(DOCS / "confusion_matrix.png", dpi=110); plt.show()

# %% [markdown]
# ## 6. Export for the browser (ONNX) and check it matches PyTorch

# %%
import onnxruntime as ort

model.eval()
onnx_path = WEB_MODEL / "fasal_doctor.onnx"
torch.onnx.export(model, torch.zeros(1, 3, IMG, IMG), onnx_path, input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
x, _ = next(iter(loaders["test"]))
with torch.no_grad():
    ref = model(x[:32]).argmax(1).numpy()
got = ort.InferenceSession(str(onnx_path)).run(None, {"image": x[:32].numpy()})[0].argmax(1)
print(f"ONNX {onnx_path.stat().st_size / 1e6:.1f} MB, agrees with PyTorch on {(ref == got).mean() * 100:.0f}% of 32 test images")

labels_out = [{"folder": f, "crop_hi": CLASSES[f][0], "disease_hi": CLASSES[f][1], "crop": CLASSES[f][2],
               "disease": CLASSES[f][3], "healthy": "healthy" in f.lower()} for f in FOLDERS]
(WEB_MODEL / "labels.json").write_text(json.dumps(labels_out, ensure_ascii=False, indent=1), encoding="utf-8")
metrics = {"test_accuracy": round(float(test_acc), 4), "test_images": int(len(ys)), "test_accuracy_exact_leaf_ids": round(float(exact_acc), 4), "classes": len(FOLDERS),
           "train_images": int(len(idx["train"])), "leaves": len(set(leaves)), "split": "by leaf (70/15/15); exact leaf IDs for 62% of photos, photo-number blocks for the rest",
           "hardest": {CLASSES[f][2] + " " + CLASSES[f][3]: round(a, 3) for f, a in sorted(per_class.items(), key=lambda kv: kv[1])[:3]},
           "history": history}
(DOCS / "metrics.json").write_text(json.dumps(metrics, indent=1))
(WEB_MODEL / "metrics.json").write_text(json.dumps({k: metrics[k] for k in ["test_accuracy", "test_images", "classes", "split"]}))
metrics
