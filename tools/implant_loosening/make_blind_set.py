"""Build an anonymised double-blind test package from the ResNet50 validation split.

The split is reproduced exactly as tools/implant_loosening/train.py computes it
(seed 42, test_ratio 0.2, val_ratio 0.15, folds 5), so the images below are the
fold1 validation images -- the split the deployed checkpoint
(server/py/weights/classifier.pt) was selected on and never trained on.
"""
import csv, hashlib, io, random, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path("tools/implant_loosening").resolve()))
from data import CLASS_NAMES, find_data_root, holdout_split, list_samples, stratified_folds

from PIL import Image

SEED = 42
N_PER_CLASS = 6
SHUFFLE_SEED = 20260923
ROOT = Path(".").resolve()
OUT_ZIP = ROOT / "blind_test_set.zip"
OUT_KEY = ROOT / "blind_test_key.csv"

root = find_data_root(ROOT / "datasets")
samples = list_samples(root)
labels = [s.label for s in samples]

pool_index, holdout_val_index, test_index = holdout_split(labels, 0.15, 0.2, SEED)
pool_all = sorted(pool_index + holdout_val_index)
pool_labels = [labels[i] for i in pool_all]
folds = [[pool_all[i] for i in fold] for fold in stratified_folds(pool_labels, 5, SEED)]
val_index = folds[0]  # fold1 == the deployed checkpoint's validation split

print(f"[data] {len(samples)} images, pool={len(pool_all)}, test={len(test_index)}")
print(f"[split] fold1 val = {len(val_index)} images")

rng = random.Random(SHUFFLE_SEED)
picked = []
for label, name in enumerate(CLASS_NAMES):
    candidates = [i for i in val_index if labels[i] == label]
    chosen = rng.sample(candidates, N_PER_CLASS)
    picked.extend(chosen)
    print(f"[pick] {name}: {N_PER_CLASS} of {len(candidates)}")

rng.shuffle(picked)

key_rows, members = [], []
for position, index in enumerate(picked, 1):
    sample = samples[index]
    case_id = f"case_{position:02d}"
    # Re-encode from raw pixels so no PNG text chunk, EXIF or original
    # filename survives into the anonymised copy.
    with Image.open(sample.path) as source:
        grey = source.convert("L")
        clean = Image.new("L", grey.size)
        clean.putdata(list(grey.getdata()))
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    members.append((f"{case_id}.png", payload))
    key_rows.append({
        "case_id": case_id,
        "true_label": CLASS_NAMES[sample.label],
        "true_binary": sample.label,
        "source_file": sample.path.name,
        "split": "fold1_val",
        "sha256": hashlib.sha256(payload).hexdigest()[:16],
    })

answer_sheet = "case_id,reader_call,confidence_0_100,notes\n" + "".join(
    f"{row['case_id']},,,\n" for row in key_rows
)
readme = f"""Blind reading set -- hip implant radiographs
{len(key_rows)} anonymised images: case_01.png .. case_{len(key_rows):02d}.png

Task: for each case, decide Loose (aseptically loose implant) or Control
(well-fixed implant). Record your call in answers.csv:

  reader_call        Loose | Control
  confidence_0_100   how sure you are, 0-100
  notes              optional

Filenames, class labels, capture order and all image metadata have been
removed; the case order is a random shuffle. The class balance is NOT
stated -- do not assume it is even. Do not open the answer key until every
reader has submitted answers.csv.
"""

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for name, payload in members:
        # Fixed timestamp: the original mtimes would leak the class grouping.
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, payload)
    for name, text in (("README.txt", readme), ("answers.csv", answer_sheet)):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, text)

with OUT_KEY.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
    writer.writeheader()
    writer.writerows(key_rows)

print(f"[done] {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.0f} KB)")
print(f"[done] {OUT_KEY}")
