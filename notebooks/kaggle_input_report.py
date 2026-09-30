"""Paste this as a single Kaggle cell to report exactly what is attached.

Prints the input tree, then checks it against what oracle_routing.ipynb needs,
so the output alone says whether a run will succeed. Stdlib only, no GPU, runs
in seconds.
"""
import os
from pathlib import Path

ROOT = Path("/kaggle/input")
SLUGS = ["gps", "gps_radar", "gps_lidar", "gps_radar_lidar", "gps_image",
         "gps_image_lidar", "gps_radar_image", "gps_radar_image_lidar"]
SOURCES = ("development", "adaptation", "test")


def walk(base):
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        yield Path(dirpath), dirnames, filenames


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


if not ROOT.exists():
    raise SystemExit("/kaggle/input does not exist -- nothing is attached.")

datasets = sorted(p for p in ROOT.iterdir() if p.is_dir())
print(f"{'=' * 78}\nATTACHED DATASETS ({len(datasets)})\n{'=' * 78}")
for ds in datasets:
    size = n = 0
    for _d, _dn, fn in walk(ds):
        for f in fn:
            try:
                size += (_d / f).stat().st_size
                n += 1
            except OSError:
                pass
    print(f"\n  {ds.name}    {human(size)}, {n} files")

# ---------------------------------------------------------------- tree
print(f"\n{'=' * 78}\nSTRUCTURE (directories, with file counts; depth <= 5)\n{'=' * 78}")
for ds in datasets:
    print(f"\n{ds.name}/")
    for d, dirnames, filenames in walk(ds):
        rel = d.relative_to(ds)
        depth = len(rel.parts)
        if depth > 5:
            dirnames[:] = []
            continue
        # collapse scenario-level leaf dirs, which are numerous and uninteresting
        if depth and rel.parts[-1].startswith("scenario") and depth >= 2:
            dirnames[:] = []
        pad = "  " * (depth + 1)
        label = "." if rel == Path(".") else rel.parts[-1]
        note = f"  [{len(filenames)} files]" if filenames else ""
        print(f"{pad}{label}/{note}")

# ---------------------------------------------------------------- findings
print(f"\n{'=' * 78}\nWHAT MATTERS\n{'=' * 78}")

indexes = [d for d, _dn, fn in walk(ROOT) if "samples.csv" in fn]
print(f"\nsamples.csv found in {len(indexes)} place(s):")
for d in indexes:
    sib = sum(1 for s in SOURCES
              for r in (d.parent, d.parent.parent, d)
              if (r / s).is_dir() or (r / s / s).is_dir())
    print(f"   {sib}/3 sources alongside   {d}")
if len(indexes) > 1:
    print("   (more than one -- the notebook picks the copy co-located with the tensors)")

print("\nscenario tensor directories:")
found_src = {}
for d, dn, _fn in walk(ROOT):
    if d.name in SOURCES and any(x.startswith("scenario") for x in dn):
        found_src.setdefault(d.name, d)
for s in SOURCES:
    if s in found_src:
        scns = sorted(x.name for x in found_src[s].iterdir() if x.is_dir())
        print(f"   {s:<12} {found_src[s]}   {scns}")
    else:
        print(f"   {s:<12} NOT FOUND")

print("\ncheckpoints:")
ckpt = {}
for d, _dn, fn in walk(ROOT):
    if "best.pt" in fn and d.name in SLUGS:
        ckpt.setdefault(d.name, d / "best.pt")
    for f in fn:
        if f.endswith(".pt") and f[:-3] in SLUGS:
            ckpt.setdefault(f[:-3], d / f)
for s in SLUGS:
    if s in ckpt:
        print(f"   {s:<24} {human(ckpt[s].stat().st_size):>9}  {ckpt[s]}")
    else:
        print(f"   {s:<24} {'':>9}  -- missing")

other_pt = sorted({str(d / f) for d, _dn, fn in walk(ROOT) for f in fn
                   if f.endswith(".pt") and f[:-3] not in SLUGS and f != "best.pt"})
if other_pt:
    print("\nother .pt files (not recognised as a configuration):")
    for p in other_pt[:10]:
        print(f"   {p}")

# ---------------------------------------------------------------- verdict
print(f"\n{'=' * 78}\nVERDICT for oracle_routing.ipynb\n{'=' * 78}")
ok = True
if not indexes:
    print("  FAIL  no samples.csv -- attach the preprocessed dataset")
    ok = False
elif "development" not in found_src:
    print("  FAIL  no development tensors -- attach the preprocessed dataset")
    ok = False
else:
    print("  OK    preprocessed dataset present")

for tier, name in (("gps", "cheap tier"), ("gps_image", "expensive tier")):
    if tier in ckpt:
        print(f"  OK    {name} checkpoint present ({tier})")
    else:
        print(f"  FAIL  {name} checkpoint MISSING ({tier}) -- the notebook cannot run")
        ok = False

print(f"  {'OK   ' if len(ckpt) == 8 else 'WARN '} {len(ckpt)}/8 checkpoints"
      + ("" if len(ckpt) == 8 else f" -- missing "
         f"{', '.join(s for s in SLUGS if s not in ckpt)}; "
         f"the two tiers are enough for the core result"))

print(f"\n  => {'READY TO RUN' if ok else 'NOT READY -- see FAIL above'}")
