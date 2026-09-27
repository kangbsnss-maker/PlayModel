# 학습 후 빠른 메뉴 인식

현재 메뉴 관찰은 숨김 PowerShell 작업자와 WinRT OCR 엔진을 재사용한다. 매 화살표 입력마다 엔진을 새로 시작하지 않는다. 같은 파일 바이트를 2초 안에 다시 관찰하면 원본 출처를 남기는 정확 일치 캐시를 사용한다. 다른 가격·문구·포커스·팝업으로 픽셀이 바뀌면 다시 읽는다. 캐시는 학습이 아니다.

2026-09-27 저장된 실제 프레임 5장 비교: 기존 OCR 중앙값 629.967ms, 지속 작업자 91.8559ms. 첫 지속 호출은 초기화 포함 628.9774ms. 5장 모두 문자와 좌표가 같았다. 원본과 측정값은 로컬 `artifacts/menu-ocr-benchmark.json`에 있다. 캐시는 이 측정에서 껐다. 이는 OCR 구간 측정이며, 화면 캡처·게임 반응을 포함한 전체 선택 시간이나 생존 실력 개선 수치가 아니다.

## 실제 학습 모델

`menu_model.py`는 화면 BGR 색상 표본에서 `scene:focus`를 예측하는 작은 선형 softmax 분류기다. 검토된 고정 manifest로 가중치를 학습하고 checkpoint를 저장한다. 행동 모방이나 빌드 최적화 모델이 아니다. 난이도 목표 `danger_6` 등은 별도 고정 규칙으로 남는다.

```powershell
.venv/Scripts/python.exe scripts/train_menu_model.py data/menu/reviewed-manifest.json models/menu/candidate-001.json
```

manifest 형식:

```json
{
  "schema": "menu-model-reviewed-manifest-v1",
  "resolution": [1920, 1080],
  "game_build_id": "23429717",
  "language": "en",
  "samples": [{
    "session_id": "실제-원본-세션-ID",
    "split": "train",
    "frame_path": "실제-frame.png",
    "frame_sha256": "실제-파일-SHA256",
    "scene": "difficulty",
    "focus": "danger_3",
    "label_provenance": {
      "kind": "human_review",
      "reviewed": true,
      "reviewer": "실제-검토자",
      "reviewed_at": "실제-검토-시각",
      "review_ref": "실제-검토-기록"
    }
  }]
}
```

위 예시는 계약 설명이며 학습 가능한 완성 데이터가 아니다. 실제 각 label은 train/validation/test에 모두 있어야 한다. 같은 세션의 분할 중복, 같은 이미지 중복, SHA 변조, 자동 OCR 정답은 거부한다. 개발자 보정은 `developer_review`로 출처를 표시한다. 사용자의 직접 시연은 필수로 두지 않는다. 검토 라벨 수집을 자체 해결하는 자동 평가기는 아직 구현하지 않았다.

## 실행 적용

평가한 checkpoint를 `models/menu/approved.json`, 그 바이트 SHA와 검토 근거를 연결한 승인 문서를 `models/menu/approval.json`에 배치한다. 승인 schema는 `menu-model-approval-v1`이며 `checkpoint_sha256`, `approved: true`, `reviewer`, `reviewed_at`, `review_ref`를 요구한다. 단순 플래그만으로는 적용되지 않는다. validation/test 양쪽에서 각 label의 수락 사례가 있고 수락한 오분류가 0이어야 한다. 이는 작은 평가에서 통과한 최소 파일 검사로, 일반화 성능 보장이 아니다. 승인 판단에는 실제 독립 평가 자료와 미지 화면 실패 사례 검토가 필요하다.

세션 시작 시 한 번 로드한다. 승인 모델이 없거나 검증이 실패하면 로컬 OCR로 실행한다. 실행 중 가중치를 바꾸지 않는다. 낮은 confidence/margin, 학습 표본과 큰 차이, 빌드·언어·해상도 불일치는 보류한다. 거리 검사는 모든 미지 화면을 검출한다고 보장하지 않는다.

빠른 경로는 현재 난이도·일시정지 화면에서 **화살표 이동만** 허용한다. 모델의 현재 포커스와 최신 프레임의 독립 픽셀 포커스 판독이 일치해야 한다. 500ms보다 오래된 관측으로 입력하지 않는다. 입력 후 새 프레임을 관찰한다. 반복 무진행·F8·단일 입력 writer 경계는 유지한다.

Enter, 상점 가격·갱신, 성장 카드 문구·선택은 현재 화면 OCR을 거친다. 최고 난이도는 **숫자 5 오른쪽의 해골 아이콘(Nightmare)**이다. 화면에 숫자 6이 표시되는 것이 아니며 `danger_6`는 그 일곱 번째 버튼의 내부 인덱스다. 난이도 시작은 이 아이콘의 현재 포커스와 `Nightmare` 문구가 함께 확인돼야 한다. 중간 버튼에서 인식이 실패하면 그 난이도로 시작하지 않는다.

전투 이동 추론은 원래부터 메뉴 OCR과 별도 경로다. 이번 변경은 게임 속도나 이동 유지 시간을 임의로 높이지 않는다. 방향 유지·위험 시 전환 조건을 보존한다.

2026-09-27 실게임 후보002를 제한 배포했다. 실제 원본 18장(train12/validation3/test3), 세션 분리·파일/복원 픽셀 중복 0. 기존 후보001에서 열람한 평가셋은 train으로 전환하고 새 validation/test를 사용했다. 검토 라벨은 `developer_review` 출처이며 자동 OCR 정답이나 사람 행동 모방 자료가 아니다. 후보002는 validation/test 6장 모두 정확하게 수락했다. 별도 미지원 화면 9장에서는 모델 오수락·빠른 화살표 허용 0건이었다. 소수 자료의 제한 검사이며 범용 정확도나 전체 게임 실력 향상의 증거가 아니다.

배포 범위는 난이도 0·1·2 인식과 최고 난이도 방향으로의 화살표 이동이다. 모델 예측 계산 중앙값은 test에서 약 0.6305ms이며 캡처·PNG 복원·특징 추출·게임 응답은 제외한다. 무기·상점·성장 선택 모델은 이 모델에 포함되지 않는다. 후보 SHA256은 `09679c9bb8a410b3bff807172e6db5b5fd13080a84df673946b7bd0b73055d51`, 원본 manifest는 로컬 `data/menu/20260927-second/reviewed-manifest.json`, 검토는 `artifacts/menu-training-review/20260927-candidate002-runtime-review.json`에 보존한다. 실행 보고서의 `menu_recognition_metrics`는 OCR/캐시/승인 모델 경로와 관찰 지연을 구분한다.
