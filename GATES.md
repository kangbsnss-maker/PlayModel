# Preparation acceptance

OWNS: README.md, AGENTS.md, pyproject.toml, .gitignore, .gitattributes, .github/**, src/**, tests/**, scripts/**, configs/**, docs/development.md, docs/github-readiness.md, docs/preparation-report.md, docs/source-cleanup.json, data/README.md, models/README.md, artifacts/README.md, media/README.md, GATES.md

- [x] G1: All source pages analyzed and consequential decisions reviewed by gpt-6-astra with ultra reasoning.
  EVIDENCE: research_design used gpt-6-astra/ultra; all 29 originals visually inspected; page-notes and source-manifest reviewed by root, with independent visual checks on pages 1-4,12,16,24,29. Architecture, M0-M7 roadmap, three ADRs, genre/media workflow and primary sources reviewed. Preservation verifier defect fixed and independently re-reviewed; pre-delete full verification passed 21 tests and all 29 source hashes/sizes/dimensions. Leaf manual gates 2/2 met.
- [x] G2: Repository preparation utilities and documentation links pass the project verification command.
  CHECK: python scripts/verify_preparation.py
  EXPECT: PREPARATION_VERIFIED
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\0. PlayModel; path=9443762079a6/40 entries; output=Ran 21 tests in 0.411s | OK
- [x] G3: Original research PNGs deleted after G1-G2; manifest and page notes retained.
  CHECK: python scripts/verify_preparation.py --check-cleanup
  EXPECT: CLEANUP_VERIFIED
  EVIDENCE: exit=0; shell=C:\WINDOWS\system32\cmd.exe; cwd=D:\0. PlayModel; path=9443762079a6/40 entries; output=CLEANUP_VERIFIED
