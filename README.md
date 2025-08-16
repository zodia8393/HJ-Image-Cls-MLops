# HJ Image Classification MLOps — CIFAR-10 엔드투엔드 자동화

운영 환경에서 재현 가능한 이미지 분류 파이프라인을 구축했습니다. 데이터 버전 관리부터 학습, 모델 레지스트리, 서빙, 모니터링, 그리고 “운영(Registry) 모델만”을 대상으로 한 오프라인 평가까지 완전 자동화했습니다.

***

## 개요

- 분야: MLOps + 컴퓨터 비전
- 목표: CIFAR-10 이미지 분류에 대해 개발·운영 전 주기 자동화
- 핵심: 모델 버전 전환/롤백, 운영 모델 품질 검증 루프, 지표 모니터링
- 대표 성능
  - Validation Accuracy: 98%+
  - 오프라인 프로덕션 모델 평가(100샘플): Accuracy 95%

***

## 아키텍처 개요

- 데이터/학습: PyTorch, CIFAR-10(32×32), DVC로 단계별 파이프라인 관리
- 실험/모델: MLflow Tracking + Model Registry
- 서빙: FastAPI (Registry에서 최신/Stage 기반 자동 로드)
- 모니터링: Prometheus + Grafana
- 배포: Docker(옵션), 확장: Optuna(하이퍼파라미터 탐색)

***

## 기술 스택

- Framework: PyTorch, torchvision
- 모델: WideResNet-28-10 (CIFAR-10 특화)
- 최적화: SGD(momentum=0.9, nesterov), CosineLR+Warmup(5ep), weight_decay=5e-4
- 증강: RandomCrop+Flip, TrivialAugmentWide(경량), Cutout(옵션), Label Smoothing(0.1)
- 데이터 정규화: CIFAR-10 mean/std
- 실험/레지스트리: MLflow
- 파이프라인: DVC
- 서빙: FastAPI(+Uvicorn)
- 모니터링: Prometheus, Grafana
- 컨테이너(옵션): Docker/Compose

***

## 리포지터리 구조

```
HJ-Image-Cls-MLops/
│  dvc.yaml
│  params.yaml
│  requirements.txt
│
├─ src/
│  ├─ scripts/
│  │  ├─ download_data.py
│  │  ├─ preprocess_data.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ mlflow_offline_test.py           # 운영(Registry) 모델 전용 오프라인 평가
│  │  ├─ batch_predict_cifar10_random.py  # 다중 호출 + /metrics 스냅샷
│  │  └─ plot_confusion_from_csv.py       # predictions.csv → 혼동행렬 PNG
│  ├─ ml/
│  │  ├─ models/
│  │  │  └─ build_model.py                # WRN-28-10 등
│  │  ├─ datamodules/
│  │  │  └─ cifar10.py
│  │  └─ utils/
│  │     └─ mlflow_utils.py
│  └─ serving/
│     └─ fastapi_app/main.py              # Registry 모델 자동 로드, /predict, /metrics
└─ outputs/ (generated)
```

***

## 파이프라인

DVC stages

- download: CIFAR-10 다운로드
- preprocess: 전처리/구조 정리
- train: 학습(AMP, 로깅, 모델 아티팩트 저장/등록)
- evaluate: 검증/지표 저장

명령

```bash
# 전체 파이프라인 실행
dvc repro
```

***

## 학습 설정

- 모델: WideResNet-28-10 (CIFAR-10 전용 구조)
- 입력: 32×32
- 증강: RandomCrop(4px pad)+Flip, TrivialAugmentWide, Cutout(옵션)
- Regularization: Label Smoothing 0.1, weight_decay 5e-4
- Optim/Schedule: SGD + Nesterov, Warmup 5ep, Cosine decay
- AMP: on
- DataLoader: pin_memory, persistent_workers로 IO 병목 완화

결과(대표)

- Validation Accuracy: 98%+
- 운영 모델 오프라인 평가(100샘플): Accuracy 95%

***

## 모델 레지스트리 운용

- 학습 완료 후 모델을 MLflow Registry에 버전으로 저장
- 승격(프로모트)
  - Staging: 검증/QA 단계
  - Production: 운영 사용 버전
- 롤백: 이전 버전을 운영 Stage로 재승격
- Stage가 비어있거나 불가한 경우에도 오프라인 평가 스크립트가 “최신 버전”으로 자동 폴백 로드

운영 연계 권장

- 서빙/배치가 Stage 기반 URI(models:/NAME/Production)를 사용하도록 구성 → 무중단 교체/롤백 용이

***

## 서빙

FastAPI 엔드포인트

- GET /health: 상태 체크
- POST /predict: 파일 업로드 → 예측 반환
- GET /metrics: Prometheus 포맷 지표 노출
  - http_requests_total, http_request_duration_seconds
  - http_requests_in_progress
  - predicted_class_total{class_id} (예측 분포)

추론 레이턴시

- 단건 평균 ≈0.14초(샘플 기준)

로컬 실행

```bash
uvicorn src.serving.fastapi_app.main:APP --host 0.0.0.0 --port 8000 --reload
```

***

## 모니터링

- Prometheus: /metrics 스크랩
- Grafana: 대시보드 권장 위젯
  - 요청 수/에러율
  - p95/p99 레이턴시
  - predicted_class_total 분포/시간 추이(드리프트 징후)

***

## 오프라인 평가(운영 모델 전용)

운영(Registry) 모델만 대상으로 재학습 없이 즉시 테스트하고, 결과를 MLflow에 기록합니다.

스크립트

- src/scripts/mlflow_offline_test.py
  - Stage 지정(예: Production), 없으면 최신 버전으로 자동 폴백
  - metrics: accuracy, tested_samples
  - artifacts: predictions.csv, sample_predictions.json

예시 실행

```bash
python src/scripts/mlflow_offline_test.py \
  --tracking-uri http://localhost:5000 \
  --model-name imgcls-resnet \
  --stage Production \
  --image-dir src/data/tests/images \
  --img-size 32 \
  --experiment offline-model-test-YYYYMMDD \
  --run-name production_eval_YYYYMMDD_HHMMSS
```

결과 예

- Resolved model: models:/imgcls-resnet/4
- Metrics: accuracy=0.95, tested_samples=100

***

## 혼동행렬(Heatmap)

- 스크립트: src/scripts/plot_confusion_from_csv.py
  - 입력: predictions.csv
  - 출력: reports/confusion_matrix.png (타이틀에 accuracy/n 표기)
  - 옵션: MLflow에 아티팩트 로깅

예시

```bash
python src/scripts/plot_confusion_from_csv.py \
  --csv offline_test_outputs/predictions.csv \
  --outdir reports \
  --log-mlflow \
  --tracking-uri http://localhost:5000 \
  --experiment offline-model-test-YYYYMMDD \
  --run-name confusion_YYYYMMDD_HHMMSS
```

***

## 대량 호출 & 모니터링 연동

- 스크립트: scripts/batch_predict_cifar10_random.py
  - 테스트 폴더에서 랜덤 N장 선택 → /predict 연속 호출
  - 호출 전/후 /metrics 스냅샷 → summary.json 생성
  - predicted_class_total 변화, 평균 지연 추정 등 기록

예시

```bash
python scripts/batch_predict_cifar10_random.py \
  --base-url http://localhost:8000 \
  --image-dir src/data/tests/images \
  --concurrency 4 \
  --outdir batch_results
```

***

## MLflow UI

로컬 실행

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

확인 포인트

- Experiments: 학습/평가 Run, 하이퍼파라미터/지표/아티팩트
- Models: 버전 목록, 승격(프로모트)/롤백, 모델 비교

***

## 자주 겪는 이슈와 해결

- 실험 이름 충돌(Deleted 상태)
  - UI 복구/영구 삭제, 또는 새로운 실험 이름 사용
- Stage 미지정 시 로드 오류
  - 오프라인 평가 스크립트에 “Stage → 없으면 최신 버전” 폴백 로직 포함
- ModuleNotFoundError(src)로 모델 로드 실패
  - PYTHONPATH에 프로젝트 루트 추가 또는 state_dict 방식 저장 권장
- 의존성 경고
  - requirements 기록에 맞춰 버전 동기화, 필요 시 환경 파일로 설치

***

## 빠른 재현(3줄)

```bash
# 1) 데이터~학습~평가
dvc repro

# 2) MLflow UI
mlflow ui --port 5000

# 3) API 기동
uvicorn src.serving.fastapi_app.main:APP --host 0.0.0.0 --port 8000
```

보너스

```bash
# 운영 모델 오프라인 테스트
python src/scripts/mlflow_offline_test.py --tracking-uri http://localhost:5000 --model-name imgcls-resnet --stage Production --image-dir src/data/tests/images --img-size 32 --experiment offline-model-test-YYYYMMDD
```

***

## 성과

- Validation Accuracy 98%+
- 평균 추론 지연 ≈0.14초(단건)
- 모델 승격/롤백 절차 정립, 운영 예외 상황에 대한 해결 가이드 축적
- 운영 모델 대상 오프라인 평가 루프 정착(회귀 방지)

***

## 다음 로드맵

- 데이터/모델 드리프트 자동 감지 + 카나리 배포·자동 롤백
- A/B 실험 자동화(두 버전 동시 서빙/로그 비교)
- 혼동행렬·클래스별 PR/F1 자동 산출 및 MLflow 기록
- 컨테이너라이즈(Compose/K8s) 및 CI/CD(프로모션·롤백 자동화)
- 모델 시그니처·입출력 유효성 검사 강화

***

## 라이선스

- 연구/교육 목적 사용 예시
- 각 외부 라이브러리는 각자 라이선스를 따릅니다

***

## 연락

- 이슈/개선 제안 환영
- 운영 환경 적용·확장(하이퍼파라미터 탐색, 멀티모델, 클라우드 배포) 지원 가능
