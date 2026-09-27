# Brotato 첫 이동 학습 파일럿

`playmodel.learning.movement`는 **작은 로컬 정책의 추론·검증된 회차 기반 가중치 갱신·checkpoint**를 구현한다. 표준 라이브러리만 사용하고 GPT/API·NumPy가 필요 없다. 구현과 합성 테스트만으로 실제 게임 학습·생존 개선·성장/상점 전략의 완성을 주장하지 않는다.

## 모델과 첫 업데이트

320×180 BGRA 관측에서 8×6 격자 중심의 RGB 144개를 뽑아 [-1,1]로 정규화한다. 직전 **실제 전송한** 9방향 행동의 one-hot 9개, 목표 방향 `(dx,dy,valid)` 3개와 bias 1개를 합쳐 **현재 입력은157차원**이다. 출력은 중립·위·우상·오른쪽·우하·아래·좌하·왼쪽·좌상 순서의9개 softmax 확률이다. 좌표의 +y는 아래다. 게임 입력 어댑터가 대각선 이동과 실제 키를 처리한다.

`random()`은 seed를 기록한 작은 무작위 가중치 기준선이다. `initialized_for_collection()`은 같은 선형 모델에 **기하 규칙으로 만든 초기 가중치**를 넣는다. 초기 목표 logit은 `navigation_strength × 단위 이동 방향 · 목표 방향`이며 기본 strength는6이다. 대각선 벡터 길이를 정규화하므로 대각선이라는 이유만으로 우선하지 않는다. `baseline_origin="geometry_prior"`와 초기 강도를 checkpoint·업데이트 보고서에 보존한다. 이는 이미 학습한 수집 능력이나 BC 정답이 아니다.

목표 검출은 연결 코드가 수행한다. `navigation=(dx,dy,True)`에서 dx/dy는 픽셀 차를 [-1,1]의 원하는 이동 방향으로 정규화한 값이다. 현재 source frame에서 관찰한 목표만 사용하고, 다른 관측에서 얻었다면 원본·관측/가용 시각·검출기 버전을 별도로 기록해야 한다. 늦게 검출된 목표를 과거 특징에 채우지 않는다. 목표가 없거나 불확실하면 `None` 또는 `valid=False`로 `(0,0,0)`을 넣어 무목표 탐색을 유지한다. 목표가 있다는 선언만으로 실제 XP 아이템인지 검증되는 것은 아니다.

`sample()`은 고정 정책에서 행동을 확률적으로 선택하고 확률분포·선택 로그확률·정책 버전·입력을 함께 반환한다. 규칙이 샘플링 뒤에 행동을 덮어쓰지 않는다. 이후 REINFORCE는 목표 방향 가중치도 실제 보상으로 갱신한다. 동일 회차 동안 가중치를 바꾸지 않는다.

v3은 같은157차원 모델에 **행동 선택 전에 결정된 허용 마스크**를 추가한다. `sample(..., mask=...)`는9개 boolean tuple을 받고 허용 행동만 softmax에 포함한다. `MovementDecision.allowed_actions`에 마스크를 보존하며 학습 검증도 같은 마스크로 확률을 재계산한다. 기본값은9방향 모두 허용이다. 사후 행동 교체나 자기 출력의 BC 라벨화가 아니다.

전투 HUD는 유지되지만 캐릭터를 잠시 못 찾는 파일럿 프레임은 `NEUTRAL_ONLY_MASK`로 중립만 실제 전송·기록한다. 확률1·로그확률0이며 해당 step의 정책 경사는 정확히0이다. 목표 특징은 `(0,0,0)`이고, 재인식 후 직전 실제 행동은0으로 이어진다.0.5초 안에 재인식하지 못하면 `perception_timeout` 중단으로 고정하며 나중의 종료 화면으로 학습 가능 회차로 바꾸지 않는다. 메뉴/HUD 경계에서는 즉시 해제하고 종료를 별도로 판독한다. 중립뿐이라 전체 경사가0이면 `training_performed=False`와 `zero_gradient`를 보고하고 후보 checkpoint를 만들지 않는다.

첫 알고리즘은 terminal REINFORCE. 독립 확인된 웨이브 종료는+1, 사망은-1, gamma=1, baseline=0이다. `R × Σ(선택 행동 one-hot − 확률) × 입력`의 정책 경사를 계산하고 전체 L2 norm을 제한한 뒤 ascent한다. 회차 내 보상을 평균 제거해 모두0으로 만들지 않는다. 자기 출력은 BC 정답 라벨로 사용하지 않는다. 종료를 확인하지 못한 시간제한·중단 회차는 학습하지 않는다.

## 학습에 필요한 기록

- 고정 원본 manifest 경로·SHA256, 캡처 설정 참조, 회차·정책·입력 generation.
- 실제 전투 진입 화면과 wave_clear/death 종료 화면의 참조·SHA256·관측/가용/검증 시각. `verified=True`, `independent_of_policy=True`가 필요하다.
- 매 행동의 불변 특징, 선택확률/로그확률, 원본 프레임 참조·SHA256, 관측→가용→결정→실제 전송 시각, 실제 전송 행동·전송 원장 참조.
- 매 행동의 허용 마스크와 원본 vision 상태. 마스크는 그 결정 전에 이용 가능했던 관측에서만 만든다. 중립 제한 프레임도 생략하지 않는다.
- 사람 개입·guard 오류·시간제한 없음. 모든 행동이 동일한 정책·generation에서 나와야 하며 이전 행동 특징도 실제 전송 순서와 일치해야 한다.

`StateEvidence.origin`은 `local_detector` 또는 `developer_verified`다. 개발자가 실제 화면을 확인한 경우 후자를 기록한다. 이는 전체 자동 결과 판독기 구현이 아니다. `validate_episode()`는 제공된 원장의 일관성과 학습 자격을 검사한다. 원장의 선언만으로 픽셀 내용이나 실제 게임 반영을 독립 입증하지 않는다. 원본 화면·입력 증거의 보존과 검토는 연결 코드의 책임이다. 입력 수신 확인이 불명인 경우 `acknowledged=None`을 유지하며 게임 반영 확인으로 바꾸지 않는다.

시계는 모두 `perf_counter_ns_same_host`. 기존 진단 캡처의 `monotonic_ns` 자료는 직접 섞지 않는다. 전투·종료를 개발자가 사후 확인하는 것은 허용하지만, 그 확인 결과를 당시 정책 특징에 소급해 넣지 않는다.

## 연결 API

```python
import random
from playmodel.learning import (
    LinearMovementPolicy, extract_features, reinforce_update,
    save_checkpoint, load_checkpoint,
)

policy = LinearMovementPolicy.initialized_for_collection(seed=20260927, navigation_strength=6.0)
rng = random.Random(20260928)  # 행동 샘플링 seed도 회차 설정에 보존
features = extract_features(
    frame_bgra, 320, 180, actual_previous_action_index,
    navigation=(normalized_goal_dx, normalized_goal_dy, True),  # 목표 미확인 시 None
)
decision = policy.sample(features, rng=rng)
# 입력 어댑터가 decision.movement를 전송한 뒤 MovementStep을 기록한다.
# StateEvidence와 완결된 EpisodeRecord는 별도 기록 경로에서 확정한다.
result = reinforce_update(policy, complete_episode, learning_rate=0.01, gradient_clip=1.0)
save_checkpoint(result.policy, new_candidate_path)  # 기존 파일은 덮어쓰지 않음
reloaded = load_checkpoint(new_candidate_path)
assert reloaded.version == result.policy.version
```

`result.report`는 원본 manifest·전/후 정책 버전·return·gradient norm·실제 가중치 변경 여부·종료 판독 출처를 기록한다. 새 모델은 후보이며 `promotion_approved=False`, `performance_improvement_verified=False`다. 승인 정책의 에피소드 경계 교체와 별도 고정 평가는 연결 코드에서 처리한다. 이전 정책의 회차를 갱신된 모델에 다시 학습시키는 off-policy replay는 거절한다.

checkpoint는 JSON이며 크기·스키마·유한 가중치·정책 digest를 검사한다. 현재 스키마는 `playmodel.linear-movement.v3`, 마스크 계약은 `predecision_legal_actions_v1`이다.154차원 v1 및 마스크 계약이 없는 v2 checkpoint를 자동 변환하지 않고 거절한다. 이전 산출물은 보존한다. 새 경로에 완성 파일을 공개하고 기존 파일을 보호한다. 원본 frame·회차·checkpoint는 저장소에서 제외되는 데이터/모델 경로에 보존한다.

## 현재 검증과 한계

테스트는8방향 목표에 정렬된 초기 확률, 목표 부재 시 탐색, 목표 가중치의 실제 갱신, 양/음 보상의 선택확률 변화, 유한차분과 경사 일치, global clipping, 수치 안정 softmax, 원본 정책 불변, 잘못된 회차 거절, checkpoint 재로딩을 확인한다. 실제 게임에서 `COMBAT → 실제 행동 → 검증된 종료 → 업데이트 → checkpoint → 새 실행`이 연결됐는지는 별도 실행 기록으로 확인해야 한다. 현재 보상 계약은 웨이브 종료/사망이며, 검증된 개별 XP 수집 보상은 아직 추가하지 않았다.

2026-09-27 v2 로컬 microbenchmark: 동일한 합성320×180 BGRA 프레임·오른쪽 목표로100회 warmup 뒤1000회 `extract_features + sample`을 측정했다. p50 **0.197ms**, p95 **0.2446ms**, p99 **0.3169ms**, 최대 **0.897ms**. 캡처·목표 검출·입력 전송·OS 스케줄링 대기는 포함하지 않는다. 이 수치는 v3 또는 실제 게임의 화면→반응5ms를 입증하지 않는다.

선형 격자 정책은 첫 학습 경로 확인용이다. 적 추적·텍스트·능력치·아이템 효과·성장 선택·전략 기억을 해결하지 않는다. 한 회차의 업데이트는 정책 실력 개선의 증거가 아니다. 같은 게임 조건의 별도 고정 평가가 필요하다.
