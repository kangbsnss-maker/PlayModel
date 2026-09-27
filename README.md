# PlayModel

목표는 **모든 장르의 게임 QA 자동화, 자동 플레이를 통한 활용성 검토, 장기적으로 인간 행동 학습**이다. 게임을 반복 실행하는 데 그치지 않고, 수집한 경험으로 로컬 모델을 반복 학습하고 독립 평가로 변화를 확인한다. 전체 방향과 현재 한계는 [학습 목적](docs/vision.md)에 정리했다.

첫 게임은 Windows Steam의 **Brotato**. AI가 캐릭터 선택부터 이동·성장·상점·종료·재시작까지 수행하고, 보유 능력치와 효과를 상황에 맞게 활용하도록 만드는 로컬 프로그램이다. 사람 시연·교정은 선택 사항이며, 검증된 실행 기록을 장르 확장과 유튜브 제작에 연결한다.

**현재: 영어 Brotato에서 캐릭터 시작부터 사망 결과까지 학습판·기준 모델 평가판·후보 모델 평가판을 완료하고 PPO 가중치 갱신을 확인했다. 승리·학습 완료·실력 향상·장르 범용성은 검증하지 않았다.** 메뉴·상점 진행, OBS 녹화와 편집·별도 한국어 자막도 검증 중이다. 1920×1080 게임 캡처와 설치된 로컬 OCR을 사용한다. 대상 게임 HWND로 키를 보내며 물리 마우스·전경 창을 가져오지 않는다. 인식 실패는 제한된 재관찰 후 중단하고 검토 파일로 남긴다. 실행 중 GPT/API 호출은 없다.

## 핵심 방향

주력 구조로 **공유 CNN + GRU + 단계별 행동 출력 + PPO**를 채택했다. 기존 화면 재사용 사전학습과 실제 신경망 전투 기록의 PPO 갱신·재로드를 검증했다. 접촉·투사체 회피 실력, 강화 방법 발견, 장시간 무인 운영 안정성은 별도 검증이 필요하다. [하드웨어·실측·실행 안내](docs/guides/local-neural-learning.md), [모델 논의 HTML](docs/research/model-comparison.html).

- 게임 실행·학습·자체 플레이 평가는 로컬 처리. GPT 호출·API 토큰 과금 필수 의존성 없음. 개발용 Astra 사용은 별개.
- CNN PPO 경로는 [온라인 actor/learner](docs/decisions/0009-online-actor-learner.md)다. 학습판의 버전별 기록으로 별도 로컬 PPO 작업자가 학습하고, 검증된 후보를 전투 중 다음 행동 경계에서 적용한다. 현재 앱에서 선택한 Laya 경로는 아래의 판단·객체 영상·구매 가치 학습을 사용한다. 독립 평가판은 모델을 고정한다.
- 최신 화면·직전 실제 전송 입력 → 로컬 정책 → 대상 게임만 조작. 작은 선형 softmax 정책은 비교 기준선이며, CNN+GRU는 새 실험 경로에서 실제 전투 수집·PPO 학습을 수행한다.
- 백그라운드 전용. 시스템 마우스·전경 창은 사용자가 유지한다. 포커스 강탈·전역 키/마우스 입력으로 자동 대체하지 않는다.
- 능력치·텍스트가 필요한 게임은 로컬 HUD/OCR, 상태 변화 추적, 게임별 의미 해석·목표 기억을 추가하고 인식 정확도와 실제 과제 성과를 각각 검증한다.
- 초기 규칙 정책·관측 지식·학습된 정책을 구분한다. 사람 자료가 있을 때 BC/교정을 선택할 수 있으며, 사람 플레이를 필수로 요구하지 않는다.
- 캐릭터 특성과 잠긴 캐릭터의 해금 조건은 원문·화면·시점과 함께 확인한다. 해금은 부목표이며 조건 진행과 실제 해금 완료를 별도로 검증한다.
- 고정 평가에서 성공률·사람 개입·기존 과제 유지율 확인 후 모델 승격.
- 장르별 게임 어댑터·평가·정책을 분리. 첫 모델을 범용 게임 AI라고 가정하지 않는다.
- 학습용 프레임과 유튜브 고화질 녹화를 분리하되 세션 ID·시간축으로 연결.

## 문서

**Laya 판단 학습:** 최근 화면을 읽는 CNN·GRU와 후보별 판단 head를 연결했다.
실제 전송과 확인된 전투 결과로 시각 문맥·판단 가중치를 갱신하며 텍스트 encoder는 고정한다.
캐릭터·시작 무기·목표 의도 순환과 UI 관측 대기를 지원한다. 나무·과실의 의미 분류와
다른 게임에서의 학습 전이 성능은 아직 검증하지 않았다.
[실행·증거 안내](docs/guides/laya.md), [화면 기반 판단 구조](docs/decisions/0011-visual-situation-decisions.md).
설치·가중치 갱신을 이동 속도나 게임 실력 향상으로 해석하지 않는다.

현재 앱의 Laya 실행은 객체 주변 crop 학습, 빠른 충돌 회피 중재, 전투 요약을 이용한
구매 가치 회귀와 UI 전이 최단경로를 함께 사용한다. 기존 CNN PPO와 별도 경로다.
HP 막대·수집·경계 후보는 관측/추정이며 확인된 처치·획득 보상으로 취급하지 않는다.
[추가 구조·학습 계약·검증 범위](docs/decisions/0013-combat-economy-implementation.md).

[전체 학습 커리큘럼: 학습판·평가판·예상 시간](docs/guides/learning-curriculum.html). 단계별 구현 상태와 통과 기준, 영상 제목 구분을 포함합니다.

**수동 실행:** 루트의 `PlayModel.vbs`를 더블 클릭한다. 시작·안전 중지·상태 확인 창을 제공한다. 창을 닫아도 별도 로컬 학습 프로그램은 계속 실행된다. [토큰 없이 실행·재시작하는 가이드](docs/guides/standalone-learning.html).

| 문서 | 내용 |
|---|---|
| [Brotato 전체 자동 플레이](docs/games/brotato.md) | 캐릭터·전투·성장·상점·결과·재시작의 전체 범위 |
| [학습 녹화·편집·녹음](docs/games/brotato-recording.md) | OBS 전 과정 녹화, 핵심 컷, 제거 가능한 한국어 자막, 스타일 설정 |
| [지속 실행·백그라운드 학습](docs/games/brotato-continuous-learning.md) | 무한 모드 반복, 별도 학습 작업자, 제목 메타데이터, 이동 보정, 미구현 경계 |
| [백그라운드 입력·스크롤 인식](docs/games/background.md) | 실제 확인 범위, 입력 분리, 메뉴 위치·끝 판단 |
| [현재 실행 안내](docs/guides/first-survivor-session.md) | 실제 캡처, 캐릭터 특성·해금 관찰, 다음 연결 단계 |
| [카드 선택·성장 학습](docs/guides/survivor-card-learning.md) | 현재 빌드와 후보에 따른 선택, 지연 효과와 비교 평가 |
| [준비 결과](docs/preparation-report.md) | 완료 범위·검증·원본 정리·다음 착수 |
| [연구 타당성 검토](docs/research/review.md) | 채택·수정·보류 판단과 장비 제약 |
| [Pluto·상태 인식·자체평가 재검토](docs/research/pluto-and-state-review.md) | 공개 범위, 능력치·텍스트 처리, 로컬 평가의 한계와 확장 |
| [전체 게임 모델 비교·논의](docs/research/model-comparison.md) | 현재 모델, Pluto, AlphaStar, PPO, DreamerV3의 차이와 후속 구조 후보 |
| [연구 페이지별 메모](docs/research/page-notes.md) | 제공 이미지 29장 전체 분석 기록 |
| [원본 출처 manifest](docs/research/source-manifest.json) | 원본 파일명·페이지·SHA256·메모 대응 |
| [1차 출처](docs/research/sources.md) | 논문·공식 문서 확인 기록 |
| [아키텍처](docs/architecture.md) | 프로그램 구성·학습·입력·데이터·영상 경계 |
| [개발 로드맵](docs/roadmap.md) | 단계별 구현·착수 조건·수락 기준 |
| [개발 시작](docs/development.md) | 설치·CLI·검증·설정 형식 |
| [GitHub 준비](docs/github-readiness.md) | 공개 범위·제외 데이터·첫 게시 절차 |
| [프로젝트 작업 규칙](AGENTS.md) | 최고 Astra 검토와 개발 원칙 |

## 빠른 확인

Python 3.12 권장. 준비 도구는 Python 3.11 이상, ML 라이브러리 설치 불필요. 공개 CLI는 `doctor`, `validate-profile`, `brotato-inspect`, `brotato-capture`다.

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e .
.venv/Scripts/python.exe -m playmodel doctor
.venv/Scripts/python.exe -m playmodel brotato-inspect
.venv/Scripts/python.exe -m playmodel validate-profile configs/profiles/brotato.template.json --allow-draft
.venv/Scripts/python.exe scripts/verify_preparation.py
```

실행 중인 Brotato의 client 한 장은 `.venv/Scripts/python.exe -m playmodel brotato-capture`로 저장한다. 게임 입력이나 학습을 수행하는 명령은 아니다. 템플릿의 미검증 항목은 `null`이며 `--allow-draft`는 형식 검사만 통과시킨다. `runtime_ready: false`를 유지한다.

## 폴더

```text
src/playmodel/       진단·캡처·백그라운드 제어·인식·로컬 학습 코드
tests/              동작·경계 검증
scripts/            준비 검증과 제한된 개발용 실게임 실험
configs/profiles/   공유 가능한 게임 프로필 템플릿
configs/local/      개인 게임·장비 설정 (Git 제외)
docs/research/      29쪽 연구 메모·출처·타당성 검토
docs/decisions/     중요 설계 결정
data/               시연·교정·고정 데이터셋 (Git 제외)
models/             후보·승인 모델 (Git 제외)
artifacts/          진단·평가·실행 산출물 (Git 제외)
media/             고화질 녹화·편집·출력 (Git 제외)
.github/workflows/  준비 검사 CI
```

## 다음 구현

최신 관측으로 이동하고 입력을 즉시 해제하는 전투 파일럿을 검증한다. 실제 전송 기록과 독립 확인된 웨이브 종료/사망을 연결해 첫 로컬 파라미터 갱신을 시험하며, 잘못 읽은 화면·개입·시간제한·제어 실패는 학습 성공으로 처리하지 않는다. 색상·위치 휴리스틱과 초기 방향 선호는 학습 성과가 아니다.

캐릭터 특성·해금 조건은 키보드 탐색과 안정된 설명 패널을 연결해 수집한다. 메뉴 스크롤과 전투 화면 이동을 구분하고, 보상·상점·다음 웨이브·전체 판·재시작을 차례로 연결한다. 전체 능력치를 고려한다는 목표를 모든 스탯의 양수화나 최적 빌드 보장으로 해석하지 않는다.

코드·문서는 [GitHub 저장소](https://github.com/kangbsnss-maker/PlayModel)에 게시한다. 게임 파일·ROM·학습 원본·모델·영상·토큰은 저장소에 포함하지 않는다. 실행과 학습은 로컬이다.

제공된 연구 이미지 29장은 전체 분석과 보존 검증 후 사용자 요청대로 삭제했다. [페이지 메모](docs/research/page-notes.md)와 [삭제 기록](docs/source-cleanup.json)을 남겼다.
