# 개발 시작

## 현재 실행 가능한 범위

CLI는 `doctor`, `validate-profile`, `brotato-inspect`, `brotato-capture`다. Brotato 설치 진단과 실제 client 1920×1080 캡처를 확인했다. 캐릭터 호버·중국어 OCR·선택 연결은 구현 중이며, GUI·정책 훈련·전체 자율 플레이·편집 영상 출력은 아직 없다. 개발자의 메뉴 교정과 학습된 모델 실행은 구분한다.

## Windows 환경

PowerShell, Git, Python 3.12 권장. 이 PC에는 uv 관리 Python 3.12.13이 설치되어 있어 재설치가 필요 없다.

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e .
.venv/Scripts/python.exe -m playmodel doctor
.venv/Scripts/python.exe -m playmodel brotato-inspect
.venv/Scripts/python.exe -m playmodel validate-profile configs/profiles/brotato.template.json --allow-draft
.venv/Scripts/python.exe scripts/verify_preparation.py
```

uv가 없는 다른 PC에서는 `py -3.12 -m venv .venv`와 `.venv/Scripts/python.exe -m pip install -e .`을 사용한다. Linux/macOS의 가상환경 Python 경로는 `.venv/bin/python`. 초기 게임 실행 호스트는 Windows를 대상으로 검토하며 다른 OS에서 준비 CLI가 통과한다고 게임 어댑터가 지원되는 것은 아니다.

설치 없이 준비 검증만 실행하려면 Python 3.11 이상에서 `python scripts/verify_preparation.py`. 검증 스크립트가 `src`를 경로에 추가하므로 pip 설치가 필요 없다.

## 종료 코드

| 명령 | 0 | 1 | 2 |
|---|---|---|---|
| doctor | 진단 완료 | 파일·실행 오류 | Python 지원 범위 미충족 |
| validate-profile | 설정 항목 완료 또는 명시적으로 draft 허용 | 형식 오류 | 정상 형식이나 미정 항목 있음 |
| brotato-inspect | 설치 파일 존재 | 파일·실행 오류 | 설치 파일 미발견/불완전 |
| brotato-capture | 관측 파일 저장 | 창·캡처·파일 오류 | 설치 파일 미발견/불완전 |

`runtime_ready`는 현재 `false`. 캡처는 검토 대기이며 `usable_for_training: false`다. GPU 발견·설정 검사·캡처 성공은 학습 성능의 증거가 아니다. doctor 출력에는 토큰·사용자명·홈 경로를 넣지 않는다.

## 실행 중인 Brotato 관찰

```powershell
.venv/Scripts/python.exe -m playmodel brotato-capture
```

기본 출력은 `data/raw/brotato-observations`다. `--steam-root`로 Steam 폴더, `--output`으로 출력 폴더를 지정할 수 있다. 대상 실행 파일·창을 확인하고 client 영역만 캡처한다. 원본과 manifest를 함께 보관하며 렌더 최신성·내용을 검토하기 전에는 학습 승인하지 않는다. 이 명령은 클릭·키 입력·훈련을 수행하지 않는다.

현재 Windows에 로컬 중국어 OCR 언어팩 `zh-Hans-CN`이 확인되었다. 언어팩 존재와 Brotato 특성의 정확한 해석은 별개다. 호버 결과는 원문·ROI·시각·추출기·판독/보정 출처를 기록하고 미해석 조건을 남긴다. 캐릭터 조건을 저장하는 작업은 관측 지식 구축이며 신경망 학습 완료가 아니다.

## 게임 프로필

`configs/profiles/brotato.template.json`을 `configs/local/brotato.json`으로 복사한다. 확정 목표는 Brotato 전체 자동 플레이이며 `adapter`에는 실제 검증한 캡처·입력·정상 재시작 경로, `evaluation`에는 고정 평가 조건을 기록한다. 미검증 항목을 추정으로 채우지 않는다. 구조 검사와 실제 준비 완료는 별도로 판단한다.

320×180, 10Hz, watchdog 250ms는 초기 측정용 제안이다. 실제 게임과 장비에서 지연·작은 HUD 인식·입력 유지 시간을 측정해 바꾼다. 해상도·Hz·전처리를 바꾸면 관련 버전을 갱신한다. watchdog 설정 검사는 watchdog 구현·작동의 증거가 아니다.

학습 프레임과 유튜브 고화질 영상의 품질·보존 정책을 따로 설정한다. 초기에 PyTorch·CUDA·PySide6·캡처 드라이버를 일괄 설치하지 않는다. 준비 코어는 표준 라이브러리만 사용한다.

## 다음 구현 단위

합성 계약과 실제 캡처를 기반으로 캐릭터 호버·특성 확인→선택·게임 시작→자율 이동→보상·상점→결과·재시작을 연결한다. 해금 조건 관찰·진행·완료 확인은 부목표로 기록한다. 규칙 정책과 검증된 자율 실행 자료를 먼저 확보하고 학습·고정 평가로 이어 간다. 사람 시연이 있으면 BC/교정에 선택적으로 사용한다. 빈 인터페이스나 개발자의 수동 판독을 완성 기능처럼 표시하지 않는다.
