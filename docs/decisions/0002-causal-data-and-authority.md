# ADR 0002 — 시간 인과·조작권·불변 데이터

날짜2026-09-26. 상태: **설계 채택, runtime 구현 전**.

## 결정

관측 사용 가능 시각·결정/대리 시각·전송 시각을 분리한다. 사건은 append-only로 보존하고 정렬/검토는 파생 버전이다. 직전 최종 전송만 정책 이력에 넣고 현재 정답/미래 프레임/개입 결과 누출을 금지한다.

입력 adapter의 유일한 writer는 조작권 관리자다. 사람 인수 시 epoch를 갱신하고 오래된 AI 제안을 거부한다. 정상 인수는 상태 원자 교체, fault는 자동입력 차단과 검증된 장치 해제 요청이다. 전송/수신/게임 적용의 unknown을 유지한다.

데이터 manifest·모델 bundle은 불변 버전이다. raw·review·train/validation/test·media를 분리한다. 캐시는 고정된 encoder/전처리/투영 경계를 명시하며 학습 중인 계층의 출력을 stale cache로 쓰지 않는다. 승인 bundle은 평가 후 episode 경계에서만 교체한다.

## 이유와 영향

기록이 잘못되면 더 큰 모델도 잘못된 정답을 학습한다. 조작권을 UI 상태 하나로 처리하면 늦은 추론 결과가 인수 뒤 전송될 수 있다. 데이터/모델을 덮어쓰면 이전 성과와 퇴행의 원인을 재현할 수 없다.

대신 시계 동기화·품질 마스크·버전 migration·파일 참조 관리 구현이 필요하다. 게임 입력 지연/사람 인지 시각을 완전히 알 수 없는 경우는 uncertainty로 드러낸다. 현실 장치의 acknowledgment 기능이 늘어나면 계약을 확장하되 기존 unknown 기록을 성공으로 덮어쓰지 않는다. [계약 상세](../architecture.md), [RLDS 근거](https://github.com/google-research/rlds).
