# ADR 0009 — 플레이 도중 반영하는 로컬 PPO

2026-09-27. 사용자의 새 요구에 따라 **학습판에서** 경험 수집·별도 프로세스 최적화·같은 판의 다음 행동 계산에 가중치 반영을 허용한다. ADR 0008의 고정 평가 경로는 유지한다. CNN+GRU와 기존 PPO를 사용한다. 실행 중 외부 API는 필요 없다.

## 선택과 경계

고정 행동 정책 버전별로 처음 64개 실제 행동을 모으는 것을 초기 설정으로 제안한다. 실측 최적값은 아니다. 메뉴에서 검증된 macro도 같은 버전의 다음 전투 앞에 포함한다. 학습 sequence 32와 실제 burn-in 8, 최대 epochs 4, 기존 KL 제한을 유지한다. 수집 시 old log probability/value/mask/hidden/실제 송신 시각을 고정한다.

learner가 그 fragment로 한 번 업데이트하는 동안 actor는 기존 가중치로 계속 움직인다. 그 뒤 추가된 이전 버전 경험은 원본으로 보존하되, 새 버전 PPO에 섞지 않는다. 한 lineage에서 미완료 학습은 하나다. 후보의 source version이 현재 actor와 다르면 적용하지 않는다. [PPO 원 논문](https://arxiv.org/abs/1707.06347)의 수집 정책과 업데이트 기준을 유지하기 위한 선택이다.

[IMPALA의 V-trace](https://arxiv.org/abs/1802.01561)는 분리된 actor/learner 사이의 정책 차이를 보정한다. [APPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo)도 비동기 수집을 지원하지만, 기존 PPO의 동일 버전 검사만 제거해서 구현할 수는 없다. 비율·값 목표·버전 지연·재사용량·recurrent replay 검증을 새로 요구하므로 첫 구현에서는 도입하지 않는다.

## fragment와 bootstrap

`FullRunRecorder`의 고정 모델 검사를 완화하지 않는다. 버전마다 별도 recorder를 만들고 메뉴 controller의 recorder도 pending 결정이 없을 때 교체한다. `finish(kind='truncated')`의 기존 manifest를 online worker에서만 학습 대상으로 허용한다. 이를 전체 판 완료로 표시하지 않는다. 기존 full-run trainer의 완료 요구는 유지한다.

truncation은 마지막 실제 송신 이후의 관측에서 **이전 정책과 마지막 실제 hidden_after**로 계산한 값을 bootstrap한다. 해당 입력·hidden·시각을 별도 hash-bound 파일로 보존하고 worker가 값을 다시 계산한다. bootstrap은 행동 샘플링이나 hidden commit이 아니다. death는 기존 독립 증거와 -1/zero bootstrap, wave clear는 기존 +1만 사용한다. 미지원 피해·회복·처치 보상은 추가하지 않는다. 시간 할인은 기존 decision-to-next-observation 계약을 따른다.

각 새 버전 fragment는 hidden 0, 첫 행동 `reset=True`로 시작한다. 내부 sequence는 실제 겹친 8개 관측으로 burn-in한다. 정책 교체에서 hidden을 재해석하거나 GAE를 버전 사이로 잇지 않는다. 메뉴 선택과 전투를 모두 보존하고 phase별 학습 행 수를 기록한다.

## 적용과 안전

후보 로딩·해시·수치 검사·추론 준비는 제어 thread 밖에서 수행한다. 현재 source 일치, 유한 weights, checkpoint 재로딩, 실제 optimizer step, 최종 KL 통과를 확인한다. 이 검사는 실력 개선 증거가 아니다.

runtime은 이전 제안이 실행 중이거나 미전송인 동안 모델을 바꾸지 않는다. 다음 계산에 새 모델을 선택하고, 마지막 실제 송신 이후 fresh frame으로 시작한다. 모델 선택과 실제 첫 송신 적용을 구분한다. 첫 새 버전 제안이 폐기되면 hidden/실제 적용 기록을 commit하지 않는다. 실제 송신 성공 때 버전·reset·hidden·segment를 일치시킨다. 기존 held-input/source-age/watchdog 한도를 초기화하거나 연장하지 않는다. STOP·guard·release는 기존 규칙대로 처리하며 무입력 gap을 정책 행동으로 위장하지 않는다.

평가판은 전체 세션 동안 모델 고정, 학습 제출과 교체 금지다. 온라인 적응 결과와 고정 모델 평가 결과를 구분한다. 같은 실제 판의 모든 버전 fragment는 동일 train/evaluation 그룹에 속한다. 이전 실험 원본·탈락 후보·사용하지 않은 tail은 삭제하지 않는다.

## 구현과 검증

`online_ppo.py`는 기존 fragment manifest를 검증하고 숨김·낮은 우선순위·CPU thread 1 worker를 실행한다. immutable request에 source/manifest/code/config를 고정한다. OS 잠금과 atomic result로 중복 최적화를 막고 저장된 후보부터 복구한다. STOP·시간 한도는 worker에도 적용한다. 불완전 후보는 덮어쓰지 않는다.

runtime은 collector snapshot과 다음 행동 경계의 교체를 담당한다. coordinator는 version별 recorder/menu 연결, 한 작업 제한, 새 실제 세션의 재개·평가·사용자 상태를 담당한다. 재개 시 저장된 가중치가 현재 게임 상태나 hidden까지 복원한다는 가정은 하지 않는다.

필수 검사: 평가 거부, source/manifest 변조 거부, bootstrap 시각·값·hidden 불일치 거부, 메뉴 행 보존, PPO 단일 버전 검증, 중복 job 방지, checkpoint 저장 후 재개, STOP, KL 결과 변조 거부, stale candidate 거부, 교체 전후 제안/실제 송신/hidden 일치. 실제 실행에서는 optimizer 구간과 전투 송신 구간의 중첩, 같은 판의 새 버전 실제 송신, capture/정책 지연과 guard를 보고한다. 단위검사를 실제 동시 학습 성공으로 대신하지 않는다.
