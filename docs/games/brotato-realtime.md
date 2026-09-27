# Brotato 실시간 반응 계약

구현: `playmodel.control.realtime`. 로컬 표준 라이브러리만 사용하는 스케줄러다. **실제 고속 캡처·OS 입력·학습된 정책 연결은 아직 없다.** `brotato-capture`의 프로세스별 PrintWindow·PNG 저장 경로는 진단용이다. 이를 매 행동마다 실행하거나 OCR 완료 뒤에만 이동하면 실시간 경로가 되지 않는다.

## 실행 경로

```text
지속 캡처 → 최신 관측 1개 → 정책 작업자 1개 → 권한·기한 재확인 → 단일 입력 sink
                ├→ 별도 OCR/상태 작업자 → 시간·신뢰도 있는 보조 관측
                └→ 별도 기록/학습 경로 → 에피소드 경계에서 승인 정책 교체
```

정책에는 지금 사용할 수 있는 화면·상태만 제공한다. OCR·학습·이미지 인코딩·파일 저장을 `tick`, 정책 호출, sink 안에 넣지 않는다. CPU/GPU를 많이 쓰는 학습은 프로세스 분리와 자원 제한도 필요하다. 별도 Python 스레드만으로 GIL·GPU 경합이나 OS 지연까지 제거되지는 않는다.

- `Observation(sequence, generation, observed_at_ns, available_at_ns, payload)`: 같은 호스트의 `perf_counter_ns`(Windows QPC) 시계. `clock_domain="perf_counter_ns_same_host"`를 검증하며 기존 진단용 `monotonic_ns` 시각과 섞지 않는다. 화면 취득 시간과 가용 시간을 구분한다. 늦은 OCR 결과로 원래 프레임 시간을 덮어쓰지 않는다. `payload`는 작업 중 수정하지 않는 바이트/읽기 전용 버퍼 등이어야 한다. 최신 슬롯은 복사하지 않으며, 별도 소비자가 읽어도 관측을 빼앗지 않는다.
- `LatestObservationSlot`: 관측 1개만 유지. 중복·역순·과거 generation·미래 시간·나이 초과를 거절한다. 정책도 동시에 1개만 계산한다. 계산 중 들어온 99장이 있으면 다음 계산은 가장 최근 1장으로 건너뛴다.
- 이미 계산 중인 관측은 더 최신 프레임이 왔다는 이유만으로 무효화하지 않는다. 현재 generation이고 나이·계산 기한을 지킨 결과만 사용한다. 지속 캡처 때문에 모든 추론 결과가 폐기되는 현상을 막기 위한 이동 경로 규칙이다. 메뉴/상점/장면 변경은 `invalidate()`로 generation을 바꾸고, 게임 어댑터가 현재 action mask·대상·비용도 검증한다.
- `arm()`은 명시적 AI 권한 부여다. `disarm()`은 generation을 증가시켜 이전 결과를 취소하고 AI 입력 해제를 요청한다. `invalidate()`는 장면/에피소드 경계에서 같은 취소를 적용한다. 승인 모델 교체는 에피소드 경계에서 기존 작업자가 종료된 뒤 새 controller/policy를 연결한다.
- `tick()`은 정책 결과를 기다리지 않는다. 오래 걸리는 정책은 계산 기한에서 입력 해제·권한 취소를 유발한다. 이미 보낸 입력을 갱신하지 못하면 watchdog 또는 원본 관측 만료 중 먼저 온 조건에서 해제를 요청한다. 종료된 결과가 나중에 돌아와도 다시 입력하지 않는다. 끊을 수 없는 정책 콜백은 daemon 작업자에 남을 수 있으며 `close()`는 종료 여부를 반환한다. 점유된 정책이 남아 있으면 재무장은 거절한다.
- `InputSink.send/release`와 권한 전환은 같은 잠금으로 직렬화한다. 인수 시 이미 시작한 sink 호출은 끝나야 한다. sink는 deadline·실제 포커스·창 식별을 OS 입력 직전에 다시 확인하고, 기한 내 반환해야 한다. `release`는 AI가 보낸 입력만 해제하고 사람 입력은 보존한다. 무한 대기하는 sink를 Python이 강제로 중단하거나 물리적 중지를 보장하지 않는다. 실제 어댑터와 독립 watchdog 시험이 필요하다.

## 연결 API

```python
from playmodel.control import ControlLimits, Observation, RealtimeController
from time import perf_counter_ns

# 숫자는 측정 전 예시 예산이며 Brotato 성능/안전 보장이 아니다.
limits = ControlLimits(
    observation_age_ns=100_000_000,
    policy_budget_ns=25_000_000,
    sink_budget_ns=5_000_000,
    input_watchdog_ns=100_000_000,
)
controller = RealtimeController(local_policy, verified_input_sink, limits)
generation = controller.arm()  # 외부 어댑터가 현재 창/장면/권한을 확인한 뒤

# 지속 캡처 작업자. generation은 캡처 시작 전에 받아 해당 프레임에 고정한다.
frame_generation = controller.generation
captured_at = perf_counter_ns()
frame = capture_readonly_frame()
controller.publish(Observation(1, frame_generation, captured_at, perf_counter_ns(), frame))

# 전용 제어 actor에서 반복. 정책은 policy(observation, deadline_ns) -> action | None.
controller.tick()
# 또는 controller.run(stop_event, interval_ns=5_000_000).
# 인수 신호는 actor 대기와 별도로 controller.disarm()을 직접 호출한다.
# 종료: worker_stopped = controller.close(); False면 정책 작업자가 아직 실행 중.
```

sequence는 같은 generation 안에서 증가한다. 모든 관측은 입력 전송 직전까지 원래 generation·시각을 유지한다. 외부 sink는 다른 코드가 직접 호출하지 않는다. `None` 정책 결과는 보류·AI 입력 해제이며 새 게임 행동이나 복구 정답 라벨이 아니다.

## 검증 범위

`tests/test_realtime.py`는 합성 관측과 기록용 sink로 최신 프레임 건너뛰기, 느린 OCR/학습과 정책의 분리, 기한 전후 검사, 인수/장면 변경 중 결과 취소, watchdog 해제, sink 지연·실패 기록을 검증한다. 실제 게임 입력·비상 정지·학습·게임 성과를 검증한 것은 아니다.

`ControlEvent`의 bounded ring은 `observed/available`, 정책 시작/종료, 전송 시작/종료, 기한, 송신 보고·수신 확인을 분리한다. `game_applied`는 항상 미확인이다. 캡처→전송 나이, 정책/전송 지연, 기한 초과·폐기율은 여기서 계산할 수 있다. 이 ring은 오래된 항목을 덮어쓰는 진단용이며 세션 원장/BC 라벨의 대체물이 아니다. 보존 기록은 별도 경로로 연결한다.

실게임 수락에는 네트워크 차단 상태의 로컬 실행, 실제 캡처→이동 p50/p95/p99, 최대 입력 유지, 사람 인수, 포커스 손실·캡처 중단·정책/입력 장애, OCR·학습 부하를 켠 반복 시험이 남아 있다. 숫자 예산은 그 실측 후 확정한다.
