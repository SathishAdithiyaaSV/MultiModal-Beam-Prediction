# QUICK_RUN smoke test — not a result

`QUICK_RUN = True`: 400 training and 300 evaluation samples, against a
2,113-parameter gate over 31 features. Establishes that
`notebooks/learned_adaptive_gate.ipynb` runs end to end without error. It does.

Do not cite these numbers. See [`../README.md`](../README.md) for what they do
and do not show, and for the oracle-ordering correction applied afterwards.

**Known defect in these CSVs.** The `ORACLE (upper bound)` rows in
`operating_curves.csv` are ordered by the Top-1 criterion, so they bound Top-1
but understate the DBA ceiling (0.8620 rather than 0.9033 at 25 % escalation).
The builder has since been fixed to emit `ORACLE (Top-1 optimal)` and
`ORACLE (DBA optimal)` separately; these files predate that.
