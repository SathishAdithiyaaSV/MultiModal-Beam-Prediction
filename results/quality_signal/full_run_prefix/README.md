# Full run — but still the pre-fix notebook

`QUICK_RUN = False`, full `val` split (2,198 samples: 600 / 750 / 848 across
scenarios 32 / 33 / 34). **The notebook executed was the pre-fix copy.**
Verified directly: this run's notebook contains zero occurrences of `ds_mods`
and zero of the constant-feature guard, both of which are present in
`notebooks/modality_quality_signal.ipynb`.

So all 13 sensor-quality features are **still identically zero**, feature group
C is **still bit-identical to group B**, and the guard that would have raised
did not exist to raise. The processing-cost vs sensing-cost question remains
untested.

**What this means practically:** the Kaggle session was running yesterday's
notebook. `QUICK_RUN` was changed, but the notebook itself was not re-pulled.

## What *is* valid, and it is worth having

Everything that does not need a sensor read is now measured at full scale, and
these numbers are good.

### Individual signals — AUC for "the cheap tier is already correct"

| feature | AUC |
|---|---|
| `top1_conf` | **0.6864** |
| `margin_15` | 0.6830 |
| `margin_12` | 0.6672 |
| `top2_conf` | 0.6498 |
| `top5_mass` | 0.6402 |
| `entropy` | 0.6133 |
| `gps_step` | 0.5577 |
| `gps_step_x` | 0.5310 |
| `gps_step_y` | 0.5302 |
| `pred_beam_norm` | 0.5160 |
| `gps_radius` | 0.5041 |

`top1_conf` at **0.6864** independently reproduces the oracle study's 0.686 on
a different code path — a useful consistency check on both.

The confidence family occupies the top six places. **GPS geometry carries almost
no routing signal** (0.50–0.56): knowing where the receiver is, or how fast it
moved, says little about whether the cheap model got this sample right.

### Feature groups — held-out-scenario AUC

| group | n features | mean AUC | per fold |
|---|---|---|---|
| **A — confidence only** | 1 | **0.6952** | 0.712 / 0.703 / 0.671 |
| B — uncertainty + availability + GPS | 18 | 0.6527 | 0.687 / 0.618 / 0.653 |
| C — B + sensor quality | 31 | *0.6527 — void, identical to B* | *identical* |

**Confidence alone beats eighteen features on held-out scenarios, at full
scale.** The quick run showed the same ordering (0.650 vs 0.566) and was
dismissible as small-sample overfitting; at n = 2,198 with every fold agreeing,
it is a result. Adding uncertainty, availability and GPS features *hurts*
cross-scenario transfer.

### How this squares with the gate result

It looks like a contradiction — the learned gate beat confidence using these
same features — but the two measure different things:

- The **gate** was trained on `train` and evaluated on `val`, i.e. **in domain**,
  the same scenarios. There, richer features help.
- These **group AUCs are leave-one-scenario-out**, i.e. **out of domain**. There,
  richer features hurt and plain confidence wins.

Both can be true, and together they say something sharper than either alone:
**the extra features encode environment-specific structure.** That is the same
pattern as the camera generalisation trap, now appearing in the gate's features
rather than the model's modalities. It is worth stating explicitly in the
write-up.

Caveat: the group AUCs rank "cheap tier correct" (a classification objective),
whereas the deployed gate regresses DBA gain. The comparison is indicative, not
exact.

## Next

Re-run with the **pulled** notebook. Confirm before trusting the output:

```python
assert "ds_mods" in open("modality_quality_signal.ipynb").read()
```

or simply check that the run does **not** silently finish — the fixed notebook
raises a `RuntimeError` if any sensor-quality feature is constant.
