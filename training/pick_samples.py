"""Copy a few TEST-split leaf photos (never seen in training) to web/samples for the demo page."""
import json, random, shutil
from pathlib import Path

src = (Path(__file__).parent / "train.py").read_text(encoding="utf-8")
exec(src.split("# %%\ncounts = Counter")[0])  # sections 1-2: classes, files and the leaf split

WANT = [("Potato___Late_blight", "आलू पछेती"), ("Potato___Early_blight", "आलू अगेती"), ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "टमाटर वायरस"),
        ("Tomato___Bacterial_spot", "टमाटर धब्बा"), ("Corn_(maize)___Common_rust_", "मक्का रतुआ"), ("Corn_(maize)___Northern_Leaf_Blight", "मक्का झुलसा"),
        ("Tomato___Septoria_leaf_spot", "टमाटर सेप्टोरिया"), ("Tomato___healthy", "टमाटर स्वस्थ")]
out = ROOT / "web" / "samples"
listing = []
for folder, label in WANT:
    ci = FOLDERS.index(folder)
    pool = [f for f, y, p, e in zip(files, labels, part, exact) if y == ci and p == "test"]
    f = random.Random(7).choice(pool)
    name = f"{CLASSES[folder][2].lower()}-{CLASSES[folder][3].lower().replace(' ', '-')}.jpg"
    shutil.copy(f, out / name)
    listing.append({"file": name, "label": label, "crop": CLASSES[folder][2], "truth": folder})
(out / "samples.json").write_text(json.dumps(listing, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(listing, ensure_ascii=False))
