# 실행 수정 이력과 회귀 확인

2026-09-27. 변경 전 관련 커밋·이 문서·기존 회귀 검사를 먼저 확인한다. 같은 `controller_guard`라도 원인이 같다고 가정하지 않는다. 아래 성공 범위 이상으로 보고하지 않는다.

| 변경 | 근거·수정 | 유지할 동작 | 검증·남은 일 |
|---|---|---|---|
| `7c97248` | 전투마다 새 OCR을 준비하던 비용 제거. 세션 OCR 공유, 정책 단계별 시간·guard 관측 증거 보존 | OCR 직렬화, 최신 프레임, 기존 입력 만료 한도 | 이후 변경에서도 공유 OCR과 시간 기록 유지 |
| `92616a4` 캡처 | 20FPS 고정 대기와 전송 후 새 프레임 요구가 겹쳐 관측 간격 증가. 실제 전송 후 캡처 깨우기, 최대 40FPS | 단일 캡처, 최신 프레임, 250ms 관측 한도 | fresh-request/캡처/전투 회귀 검사 통과. 모든 지연 해결이라는 뜻 아님 |
| `92616a4` 온라인 학습 | 64개 같은 버전 행동과 후속 관측으로 별도 PPO. 검증된 후보를 실제 다음 송신 때 적용 | 미전송 후보 불반영, GRU 초기화, 고정 평가 분리, 원본 보존 | 전체 503검사 통과. 실제 전투 중 optimizer→새 버전 전송 확인은 아직 미완료 |
| `f5ba68d` OBS | 19:51:37 Windows Application 1000: OBS `obs.dll / 0xc0000005`. 녹화 연결 초기화 실패 | 다른 녹화 소유권 보호, 기존 OBS 중복 실행 금지 | 시작 전 연결 확인·없을 때 실행. 4검사 통과. OBS 내부 충돌 원인 자체를 고친 것은 아님 |
| `f5ba68d` 메뉴 재개 | OBS 중단 뒤 난이도 화면. 기존 `_new_run`은 결과/사망만 허용. 초기 복귀 수정에서 Escape 반복으로 타이틀까지 이동 | 확인된 화면에서만 입력, 캐릭터·무기 새 증거, 방향키에 일괄 대기 추가 금지 | 화면별 Escape 1회, 기존 transition gate 재사용, 시작/캐릭터/무기/난이도 재개. 반복 화면 회귀 검사 및 실제 전투 진입 확인 |

## 현재 중단 증거

`artifacts/recurrent-cycles/online-20260927T104959Z-5ab59ff8/training-4e104919fadb4658b8cb16438324bca4`

전투 32개 행동 후 `sink_deadline`. 다음 정책 계산 16.65ms, `BackgroundController.set_movement` 31.39ms로 송신 예산 25ms 초과. 앞선 `held_observation_expired`와 다른 원인이다. 이 실행에서는 PPO fragment가 아직 없으므로 학습 작업이 원인이라고 판단하지 않는다.

후속 수정: 같은 OS 실행파일 문자열의 반복 `Path.resolve`를 줄이고 identity/키매핑 후 원래 deadline을 재검사한다. 단계별 시각을 기록한다. 31.39ms 중 어느 단계가 지연됐는지는 기존 로그만으로 확정할 수 없다. PID·실행파일·F8 매번 점검과 25ms/250ms 한도는 유지했다. 관련 69검사, 시작·OBS·전체 판 계약 24검사 통과. 실제 안정성 확인을 위해 20:05:16 재개했다. 런타임 소스 hash에 background/interaction/setup_run도 포함한다.

## 수정 순서

1. 관련 파일의 `git log`와 최근 diff, 마지막 실제 실패의 guard·정책·캡처 기록을 대조한다.
2. 기존 수정으로 보장한 동작과 이번에 바꿀 부분을 명시한다. 공통 원인이 확인되기 전 새 우회 경로를 추가하지 않는다.
3. 기존 회귀 검사에 재발 조건을 추가하고, 해당 검사 묶음 통과 후 한 번 재개한다.
4. 런타임 변경 중에는 플레이하지 않는다. 재개 후 읽기 전용 로그로 확인한다.
5. 실제 optimizer와 실제 새 버전 송신을 구분해 기록한다. 새 실패는 같은 오류명만 보고 이전 수정 실패로 단정하지 않는다.

핵심 검사: `test_neural_runtime`, `test_stream_fresh_requests`, `test_online_runtime`, `test_online_ppo`, `test_full_run`, `test_recurrent_pipeline_overlap`, `test_setup_recovery`, `test_obs_startup`.
