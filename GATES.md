# Gates: combat, economy and UI learning integration

OWNS: src/playmodel/**, scripts/**, tests/**, docs/**, README.md, GATES.md

Scope: local evidence-based learning, preserving single input writer and existing weights.

Earlier preparation evidence: [preserved ledger](docs/gates/preparation-accepted.md).

- [x] G1: Object observations, HP uncertainty, motion learning, collection and boundary evidence are causal and persist.
  CHECK: .venv\Scripts\python.exe -m unittest discover -s tests -p test_combat_experience.py
  EXPECT: /^OK\r?$/m
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\0. PlayModel; path=9158a800b609/38 entries; output=Ran 6 tests in 3.476s | OK
- [x] G2: Fast risk arbitration and combat-to-shop learning are connected without false action credit.
  CHECK: .venv\Scripts\python.exe -m unittest discover -s tests -p test_combat_economy.py
  EXPECT: /^OK\r?$/m
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\0. PlayModel; path=9158a800b609/38 entries; output=Ran 9 tests in 2.364s | OK
- [x] G3: UI graph learns actual transitions and selects wrap/short paths without bypassing confirmation.
  CHECK: .venv\Scripts\python.exe -m unittest discover -s tests -p test_ui_navigation.py
  EXPECT: /^OK\r?$/m
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\0. PlayModel; path=9158a800b609/38 entries; output=Ran 6 tests in 0.039s | OK
- [x] G4: Full regression and preparation checks pass.
  CHECK: .venv\Scripts\python.exe scripts/verify_preparation.py
  EXPECT: PREPARATION_VERIFIED
  EVIDENCE: exit=0; shell=PowerShell via Invoke-Hidden; cwd=D:\0. PlayModel; output=Ran 702 tests in 74.901s | OK | PREPARATION_VERIFIED. Follow-up typed-release change: 8 scheduling recovery tests PASS and actual failure proof replay accepted; GitHub Preparation checks for bb4c590 passed. Direct full rerun followed an earlier combined gate failure without retained full failure output.
- [x] G5: Astra design/review resolves blocking issues and documentation distinguishes observations from labels.
  EVIDENCE: jev_design_review (gpt-6-astra/ultra), read-only final review: no blocking findings; evaluation recovery exclusion, model provenance, UI direction-only boundary, contextual purchase values and gap credit clearing confirmed.
- [x] G6: Reviewed code/docs are committed and remote branch matches the pushed commit; no game/model/data files staged.
  EVIDENCE: main fb44049 pushed and GitHub Preparation checks passed; bb4c590 follow-up recovery fix pushed. Staged audit: 46 code/test/doc files, 589532 bytes, no runtime/model/data paths or credential patterns. HEAD/origin/main matched after push.
- [x] G7: Authorized continuous learning resumes with preserved weights, records new structures and accepts a real update.
  EVIDENCE: live session laya-20260927T160330Z-df8f78c9: status running/combat, error null, accepted update 1; object projection delta 0.0026432689 (initial encoder gradient intentionally zero); auxiliary motion training 917 pairs, revision 1. Prior session verified actual difficulty 0-left-6 and 4 object crops per tactical choice. No exact HP/kill/pickup labels asserted.
