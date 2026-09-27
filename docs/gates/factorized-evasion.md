# Factorized evasion acceptance evidence

- [x] Environment, raw weapon observations, known stats, threat trajectories and nine destination hypotheses recorded separately.
- [x] Future observation excluded from features; gaps/context changes/broken execution chains censored; run-level split key retained.
- [x] Dataset checksum and observational predictor persisted; restored model hash matches; fixed evaluation produces no dataset/model update.
- [x] Regression verification: `scripts/verify_preparation.py`, 709 tests, exit 0, 76.271 seconds, PREPARATION_VERIFIED. PowerShell, workspace D:\0. PlayModel, hidden Python execution.
- [x] Live session laya-20260927T162022Z-52015eb1, revision 14: 65 factorized rows, 39 distance regression examples, persisted predictor samples 39. Dataset SHA256 1da9834b2637b1346033db5a2105851281b4dc573b58a8f3355a731a30c7e955. Status running, error null.

Limits: actual map geometry and projectile/weapon effects remain unknown. The auxiliary model predicts observed distance, not safe-action labels, and does not directly control movement.
