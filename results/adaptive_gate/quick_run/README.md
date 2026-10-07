# QUICK_RUN smoke test — superseded by `../full_run/`

`QUICK_RUN = True`: 400 training and 300 evaluation samples, against a
2,113-parameter gate over 31 features. Its purpose was to establish that
`notebooks/learned_adaptive_gate.ipynb` runs end to end. It does.

**Do not cite these numbers.** They concluded that the learned gate was within
noise of a confidence threshold (+0.001 DBA at 25 % escalation); the full run
shows +0.017, and the apparent null was an artefact of fitting the gate on 400
samples. See [`../full_run/`](../full_run/).

Retained for two reasons: it is the record of the execution check, and it is
what exposed the oracle-ordering bug — `regress_gain` scored 0.8271 here
against an "oracle" of 0.8236 at 10 % escalation, which is impossible and led
to the fix described in [`../README.md`](../README.md).

The `ORACLE (upper bound)` rows in this directory's `operating_curves.csv` are
Top-1-ordered and understate the DBA ceiling.
