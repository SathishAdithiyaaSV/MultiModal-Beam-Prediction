# Paste as one Kaggle cell. Finds exactly which files the dataset is missing.
import os
from pathlib import Path
import pandas as pd

ROOT = Path("/kaggle/input")
SOURCES = ("development", "adaptation", "test")
MODS = ("camera", "lidar_bev", "radar_ra_rv")

def walk(b):
    for dp, dn, fn in os.walk(b, followlinks=True):
        yield Path(dp), dn, fn

# the index co-located with the tensors
def find_source(base, name):
    for c in (base / name / name, base / name):
        if c.is_dir() and any(c.glob("scenario*")):
            return c
    return None

hits = [d for d, _dn, fn in walk(ROOT) if "samples.csv" in fn]
def score(h):
    return sum(1 for s in SOURCES
               for r in (h.parent, h.parent.parent, h) if find_source(r, s))
INDEX = sorted(hits, key=score, reverse=True)[0]
print(f"index: {INDEX}\n")

SRC = {s: next((find_source(r, s) for r in (INDEX.parent, INDEX.parent.parent, INDEX)
                if find_source(r, s)), None) for s in SOURCES}

print(f"{'source':<13}{'scenario':<12}{'modality':<14}{'on disk':>9}")
present = {}
for s, base in SRC.items():
    if base is None:
        continue
    for scn in sorted(base.glob("scenario*")):
        for m in MODS:
            d = scn / m
            n = len(list(d.iterdir())) if d.is_dir() else 0
            present[(s, scn.name, m)] = n
            print(f"{s:<13}{scn.name:<12}{m:<14}{n:>9}")

# what the index actually requires
sam = pd.read_csv(INDEX / "samples.csv")
print(f"\n{'='*70}\nrequired vs present, per source/scenario/modality\n{'='*70}")
COLMAP = {"camera": "image", "lidar_bev": "lidar_bev", "radar_ra_rv": "radar"}
MASK = {"camera": "m_image", "lidar_bev": "m_lidar", "radar_ra_rv": "m_radar"}
bad = []
for (s, scn, m), n_disk in sorted(present.items()):
    g = sam[(sam.source == s) & (sam.scenario == scn)]
    if g.empty:
        continue
    need = set()
    for k in range(1, 6):
        col = f"{COLMAP[m]}_{k}"
        need |= set(g.loc[g[MASK[m]] == 1, col].map(lambda p: Path(p).name))
    missing = len(need) - n_disk
    flag = "" if missing <= 0 else f"   <-- {missing} MISSING"
    print(f"  {s:<13}{scn:<12}{m:<14} need {len(need):>5}  have {n_disk:>5}{flag}")
    if missing > 0:
        bad.append((s, scn, m, need, n_disk))

if not bad:
    print("\nthe dataset is complete for every modality the index marks available")
else:
    print(f"\n{'='*70}\nexamples of missing files\n{'='*70}")
    for s, scn, m, need, _n in bad:
        base = SRC[s] / scn / m
        gone = sorted(x for x in need if not (base / x).exists())
        print(f"\n  {s}/{scn}/{m}: {len(gone)} missing, e.g.")
        for x in gone[:5]:
            print(f"     {x}")
