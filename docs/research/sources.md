# 1차 출처 확인 기록

확인일: **2026-09-26**. 제공 이미지의 R번호를 유지했다. 아래는 실제 확인한 논문·제작자 저장소·공식 문서다. 저장소 전체 코드 감사, 논문 재현, 장비 벤치마크를 수행했다는 뜻은 아니다. 변하는 문서의 설치 예시는 채택 시 릴리스/커밋을 별도로 고정한다.

| ID | 직접 확인한 출처 | 뒷받침하는 범위와 제한 |
|---|---|---|
| R01 | Ross et al. 2011, [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | 행동이 이후 관측 분포를 바꾸는 순차 예측과 데이터 누적 원칙. 선택적 게임 교정의 동일 이론 보증을 주장하지 않음 |
| R02 | Kelly et al., [HG-DAgger](https://arxiv.org/abs/1810.02890) | 사람 조작권 인수와 상호작용 모방. v1 2018, v2 2019. 자율주행 실험을 게임 성능으로 이전하지 않음 |
| R03 | Mandlekar et al. 2021, [robomimic 연구](https://arxiv.org/abs/2108.03298), [구현 알고리즘](https://robomimic.github.io/docs/introduction/implemented_algorithms.html), [BC-RNN 예제](https://github.com/ARISE-Initiative/robomimic/blob/master/examples/train_bc_rnn.py) | 사람 시연·순환 BC 및 평가 설계 참고. 예제는 LSTM을 사용하고 GRU도 지원. 원안 GRU512/30Hz는 공식 게임 설정이 아님 |
| R04 | [Torchvision resnet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) | ImageNet 가중치와 기본 resize/중앙 crop/정규화 확인. 본 프로젝트의 320×180/공간풀링은 변형 설계 |
| R05 | Baker et al. 2022, [VPT](https://arxiv.org/abs/2206.11795) | 소량 행동 라벨로 inverse dynamics를 학습해 대규모 무라벨 Minecraft 영상을 라벨링한 다음 BC/RL. 입력 없는 영상이 곧 정확한 행동 데이터라는 뜻이 아님 |
| R06 | [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | 온폴리시 수집·행동 로그확률·가치 학습. 문서의 구현을 게임 순환/교정 PPO 완성품으로 채택하지 않음 |
| R07 | [Google Research RLDS](https://github.com/google-research/rlds) | 에피소드·step·출처·최종 관측. `is_last`에서 action/reward 등의 유효성 규약에 유의해 변환 시험 필요 |
| R08 | [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) | MP4/Parquet/metadata, 여러 에피소드의 파일 공유와 오프셋. 저장 설계 참고이며 LeRobot 전체를 게임 런타임 필수 의존성으로 만들지 않음 |
| R09 | [Streaming Video Encoding](https://huggingface.co/docs/lerobot/en/streaming_video_encoding) | 수집 중 영상 인코딩, 큐 크기, 기록과 제어 자원 경쟁. 기본 설정/성능 수치는 우리 호스트 보증이 아님 |
| R10 | [Video encoding parameters](https://huggingface.co/docs/lerobot/en/video_encoding_parameters) | 코덱·키프레임·픽셀형식·품질의 크기/디코딩 절충 |
| R11 | [FFmpeg segment muxer](https://ffmpeg.org/ffmpeg-formats.html#segment_002c-stream_005fsegment_002c-ssegment) | 분할은 키프레임과 시간축의 영향을 받음. 요청한 절단 시각과 실제 파일 경계를 구분 |
| R12 | [Stable-Retro Replay files](https://stable-retro.farama.org/python/#replay-files) | `.bk2` 초기 상태/버튼 기록, 재생, 영상 출력 기능. 각 게임/코어/버전에서 직접 일치 검사 후 영상 대체 가능 |
| R13 | [Stable-Retro](https://stable-retro.farama.org/), [Getting Started](https://stable-retro.farama.org/getting_started/) | 에뮬레이터 기반 게임 환경 후보. 대상 게임·OS wheel·코어·초기 상태를 아직 선택/설치하지 않음 |
| R14 | [Pluto 제작자 README](https://github.com/tscmoo/pluto/blob/main/README.md) | 제작자는 BWAPI 기반 브루드 워 자가대전 모델과 바이너리 배포를 설명. 전체 학습 소스 공개/재현 가능성이나 시각 입력 정책임을 뜻하지 않음 |
| R15 | Kumar et al. 2020, [CQL](https://arxiv.org/abs/2006.04779) | 고정 자료에서 분포 밖 행동 가치를 낙관하는 오프라인 RL 문제. 본 계획은 CQL 구현을 포함하지 않음 |
| R16 | Schulman et al. 2017, [PPO](https://arxiv.org/abs/1707.06347) | 새 정책 상호작용 자료와 제한된 정책 갱신. 순환 기억/사람 인수 혼합 처리는 별도 설계 |

추가 구현/영상 출처:

| ID | 직접 확인한 출처 | 현재 설계에 적용 |
|---|---|---|
| S01 | [PyTorch 로컬 설치](https://pytorch.org/get-started/locally/) | Windows Python 지원 범위에 3.12 포함. 버전 선택기/개별 문서 표기가 서로 다른 시점일 수 있어 최신 버전 번호를 여기서 확정하지 않음 |
| S02 | [Qt for Python 시작](https://doc.qt.io/qtforpython-6/gettingstarted.html) | Python3.10+, venv 권장, PySide6 wheel에 Qt 포함. 데스크톱 셸 후보의 설치 기반 |
| S03 | [NVIDIA CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) | CUDA13 계열은 driver580 이상 요구. 현재560.94에서 CUDA13 자동 채택 불가. CUDA12도 실제 wheel/기능/장치 실행 검증 필요 |
| S04 | [Gymnasium 시간 제한](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | terminated와 truncated는 가치 bootstrap 처리가 다름. 센서 손상/사람 인수를 단순 시간제한과 같게 처리하지 않음 |
| S05 | [OBS 녹화 출력 안내](https://obsproject.com/kb/standard-recording-output-guide) | 중단 내성을 위해 MKV 녹화 후보, 별도 오디오 트랙. 출력 인코더·화질 증가는 자원을 추가 사용 |
| S06 | [YouTube 업로드 인코딩](https://support.google.com/youtube/answer/1722171?hl=en) | MP4/H.264·48kHz 오디오, 촬영 FPS 유지, SDR1080p30/60의8/12Mbps 권장. 이는 업로드 사양이며 학습 영상/원본 녹화 최적값 아님 |
| S07 | [YouTube 게임·소프트웨어 콘텐츠](https://support.google.com/youtube/answer/138161?hl=en) | 게임 퍼블리셔의 이용 허용 범위와 콘텐츠 조건 확인 필요. 장시간 단순 플레이의 수익화는 보장되지 않음. 구체적 게임의 권리는 게임 선택 후 확인 |

R14의 CPU 추론/모델 성능 설명은 **제작자 주장**으로만 취급한다. 이번 설계에서 가져오는 것은 비교 대상의 환경·인터페이스 구분이다. R08–R10의 로봇 기능, R05의 대규모 학습 자원, S06의 영상 권장값을 PlayModel의 검증된 성능으로 쓰지 않는다.
