"""Temperature scaling calibrator.

Lands in M3. Per-question-type scalar T fit with LBFGS on NLL over a held-out
set. Persisted to calibration/<model>.json and versioned. Report ECE (15 bins),
Brier, NLL — before and after.
"""
