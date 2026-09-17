"""Settle the double-preprocessing question empirically, before deploying.

The preprocessing was read out of the saved graph's structure — strong evidence
but not proof. This runs the model both ways on images whose class is known and
prints which one is actually right.

    pip install tensorflow-cpu pillow numpy

Directory mode (preferred) — truth comes from the class subfolder name:

    python verify_model.py MODEL.keras --data path/to/data --per-class 5

    data/
      black_hole/ hole_002.jpg ...
      galaxy/     ...

File mode — truth comes from the filename, which must contain the class name:

    python verify_model.py MODEL.keras saturn_01.jpg galaxy_02.jpg
"""
import argparse
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input as dn_pre
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_pre

keras.mixed_precision.set_global_policy("float32")

CLASS_NAMES = [
    "black_hole", "earth", "galaxy", "jupiter", "mars", "mercury",
    "neptune", "pluto", "saturn", "uranus", "venus",
]
IMG_SIZE = (224, 224)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(path):
    co = {"preprocess_input": dn_pre}
    try:
        return tf.keras.models.load_model(path, custom_objects=co, compile=False)
    except (TypeError, ValueError) as e:
        print(f"  (retrying with safe_mode=False: {type(e).__name__})")
        return tf.keras.models.load_model(
            path, custom_objects=co, compile=False, safe_mode=False
        )


def collect(args):
    """Return [(path, true_class_or_None), ...]."""
    if args.data:
        items, rng = [], random.Random(args.seed)
        for cls in CLASS_NAMES:
            folder = os.path.join(args.data, cls)
            if not os.path.isdir(folder):
                print(f"  ! no folder for class {cls!r}, skipping")
                continue
            files = [f for f in sorted(os.listdir(folder))
                     if f.lower().endswith(EXTS)]
            for f in rng.sample(files, min(args.per_class, len(files))):
                items.append((os.path.join(folder, f), cls))
        return items

    out = []
    for p in args.images:
        base = os.path.basename(p).lower()
        # longest match first, so "black_hole" wins over any shorter overlap
        truth = next((c for c in sorted(CLASS_NAMES, key=len, reverse=True)
                      if c in base), None)
        out.append((p, truth))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--data", help="directory of class subfolders")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true",
                    help="print every image, not just the summary")
    args = ap.parse_args()

    if not args.data and not args.images:
        print(__doc__)
        sys.exit(1)

    print(f"Loading {args.model} ...")
    model = load_model(args.model)

    names = [t.name.split(":")[0].split("/")[0] for t in model.inputs]
    print(f"  inputs : {names}")
    print(f"  output : {model.output_shape}")
    print(f"  params : {model.count_params():,}\n")

    def feed_raw(arr):
        return arr if len(names) == 1 else {n: arr for n in names}

    def feed_double(arr):
        if len(names) == 1:
            return dn_pre(arr.copy())
        return {names[0]: vgg_pre(arr.copy()), names[1]: dn_pre(arr.copy())}

    items = collect(args)
    print(f"Scoring {len(items)} image(s)...\n")

    n = raw_ok = dbl_ok = 0
    raw_conf, dbl_conf = [], []
    per_class = {}

    for path, truth in items:
        try:
            img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        except Exception as e:
            print(f"  ! skipping {os.path.basename(path)}: {e}")
            continue
        arr = np.expand_dims(np.array(img, dtype="float32"), 0)

        pr_raw = model.predict(feed_raw(arr), verbose=0)[0]
        pr_dbl = model.predict(feed_double(arr), verbose=0)[0]

        a, b = int(np.argmax(pr_raw)), int(np.argmax(pr_dbl))
        raw_conf.append(float(pr_raw[a]))
        dbl_conf.append(float(pr_dbl[b]))

        if args.verbose or not args.data:
            print(f"{os.path.basename(path):34} (true: {truth or 'unknown'})")
            print(f"   raw 0-255     -> {CLASS_NAMES[a]:12} {pr_raw[a]:6.1%}")
            print(f"   pre-processed -> {CLASS_NAMES[b]:12} {pr_dbl[b]:6.1%}")

        if truth:
            n += 1
            r, d = CLASS_NAMES[a] == truth, CLASS_NAMES[b] == truth
            raw_ok += r
            dbl_ok += d
            acc = per_class.setdefault(truth, [0, 0, 0])
            acc[0] += 1
            acc[1] += r
            acc[2] += d

    if not n:
        print("Nothing was scored — no class could be determined.")
        return

    print("\n" + "=" * 58)
    print(f"{'class':<14}{'n':>4}{'raw':>10}{'pre-processed':>16}")
    print("-" * 58)
    for cls in CLASS_NAMES:
        if cls in per_class:
            t, r, d = per_class[cls]
            print(f"{cls:<14}{t:>4}{r/t:>9.0%}{d/t:>16.0%}")
    print("-" * 58)
    print(f"{'OVERALL':<14}{n:>4}{raw_ok/n:>9.1%}{dbl_ok/n:>16.1%}")
    print(f"{'mean conf':<14}{'':>4}{np.mean(raw_conf):>9.1%}{np.mean(dbl_conf):>16.1%}")
    print("=" * 58)

    if raw_ok > dbl_ok:
        print("\n=> RAW WINS. app.py is correct as written; "
              "utils/preprocessing.py double-preprocesses.")
    elif dbl_ok > raw_ok:
        print("\n=> PRE-PROCESSED WINS. The graph does not preprocess internally — "
              "app.py must apply vgg_preprocess / densenet_preprocess before feeding.")
    else:
        print("\n=> TIED. Raise --per-class and rerun.")


if __name__ == "__main__":
    main()
