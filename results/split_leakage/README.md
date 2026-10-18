# Temporal leakage in DeepSense6G splits

This project reports **Top-1 0.4604** where the AMBER paper reports **0.6415**.
That gap is the first thing a reviewer will challenge, and until now we only
*asserted* it came from our stricter protocol. This measures it.

Produced by [`analysis/split_leakage.py`](../../analysis/split_leakage.py).
No GPU; it runs from the cached per-sample tables in seconds.

## The mechanism

Each sample is **W = 5 consecutive observations** ending at `frame_target`. Two
samples whose targets differ by `d < 5` therefore **share 5 − d of their actual
input frames** — the same JPEG, the same LiDAR sweep, the same radar cube
appearing on both sides of the split. This is not correlation between similar
scenes. It is the same data.

## What a random split does

11,042 labelled development samples, 20 % held out, averaged over 20 draws:

| | block split (ours) | random split |
|---|---|---|
| val samples **sharing input frames** with train (`d<5`) | **13.6 %** | **100.0 %** |
| val samples within 1 frame of a train sample | 3.4 % | 95.3 % |
| val samples within 10 frames | 32.7 % | 100.0 % |
| median frames to nearest train sample | **16** | **1** |

**Under a random split, every single validation sample shares input frames with
the training set.** Not most — all of them. The median validation sample sits
one frame from a training sample.

## How much the leak gives away

The optimal beam changes slowly, so a leaked neighbour hands over most of the
answer:

| separation | P(same beam) | vs chance (1/64 = 1.6 %) |
|---|---|---|
| 1 frame | **39.4 %** | **25×** |
| 2 frames | 35.8 % | 23× |
| 3 frames | 32.1 % | 20× |
| 5 frames | 26.8 % | 17× |

A sample one frame from a training example carries the *same beam label*
39.4 % of the time. Combined with near-identical inputs, a model that has
memorised its training set is handed a large fraction of the validation set.

## What this establishes

**The 0.4604 vs 0.6415 gap is explained by protocol, and we can now say so with
numbers rather than hand-waving.** A randomly-split evaluation on this dataset
is not measuring generalisation to new moments; it is substantially measuring
recall of frames the model has already seen. Our figures are lower because they
are measured on genuinely held-out time.

This is worth stating as a contribution in its own right. Random splitting is
common on DeepSense6G, and the evidence here says it inflates reported accuracy
on a dataset the community actively benchmarks against.

## Where our own split falls short

**13.6 % of our validation samples still share input frames with training
data**, and 3.4 % sit within one frame. Contiguous 50-frame blocks prevent
wholesale mixing, but every block boundary creates a handful of adjacent pairs
straddling the split.

That is far better than 100 %, but it is not zero, and we should not claim a
clean split while 1 in 7 validation samples overlaps.

**The fix is cheap: a guard band.** Dropping samples whose target lies within
`W − 1 = 4` frames of a block boundary would take the overlap to exactly 0 %, at
a cost of roughly 8 % of the data. Worth doing before submission, and worth
reporting both ways so the effect of the guard band is visible.

## Files

| file | contents |
|---|---|
| `split_leakage.json` | the full comparison, both split policies, 20 random draws |
| `beam_persistence.csv` | P(same beam) against temporal separation, 1–12 frames |
