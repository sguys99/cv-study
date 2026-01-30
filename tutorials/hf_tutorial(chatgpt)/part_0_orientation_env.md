# Part 0. 오리엔테이션 & 실습 환경

## 0.1 과정 구조/산출물/평가 기준
- 공통 흐름
  - 문제 정의 → 데이터/모델 선택 → 학습/튜닝 → 평가 → 배포/데모 → 리포트
- 모든 실습의 “완성” 정의
  - (필수) `train.py / infer.py / eval.py / serve.py` 형태로 실행 가능
  - (필수) 재현 가능한 실행 커맨드 + 결과(지표/샘플) 기록
  - (권장) Hub 업로드(모델/데이터/Spaces 중 1개 이상)

- 산출물(제출물 체크리스트)
  - 코드: 실행 가능한 스크립트 + README
  - 결과: 지표(Accuracy/F1 또는 mAP/IoU 등) + 실패 케이스 Top-N
  - 운영 관점: 비용/지연시간 1줄 요약 + 개선 아이디어 3개

- 평가 루브릭(예시)
  - 재현성 30%: 설치/실행/결과 재현
  - 엔지니어링 25%: 설정 분리, 로깅, 예외 처리, 구조
  - 평가/분석 25%: 지표 + 오류 분석 + 개선안
  - 배포/데모 20%: 로컬 API 또는 Spaces 데모

- 참고 링크
  - Transformers Quicktour: https://huggingface.co/docs/transformers/en/quicktour
  - HF Hub: https://huggingface.co/docs/hub
  - Spaces(Gradio): https://huggingface.co/docs/hub/en/spaces-sdks-gradio

---

## 0.2 개발환경 세팅 (GPU, CUDA, venv/conda, HF Token, 캐시/스토리지 전략)

### 필수 체크
- Python 3.10+ 권장
- GPU 사용 시
  - `nvidia-smi` 동작 확인
  - PyTorch는 공식 설치 셀렉터로 설치(환경/OS에 맞는 커맨드 사용)
    - https://pytorch.org/get-started/locally/

### venv
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

### conda
```bash
conda create -n hf-tutorial python=3.10 -y
conda activate hf-tutorial
python -m pip install -U pip
```

### 기본 패키지
```bash
pip install -U transformers datasets evaluate accelerate huggingface_hub
pip install -U fastapi uvicorn[standard] scikit-learn numpy
```

### HF Token
- 토큰 생성/권한: https://huggingface.co/docs/hub/en/security-tokens
- CLI 로그인:
```bash
hf auth login
```
- CLI 가이드: https://huggingface.co/docs/huggingface_hub/en/guides/cli

### 캐시/스토리지(권장)
- 큰 디스크로 캐시 고정(서버/WSL/노트북 공통)
- 환경변수 문서:
  - https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
  - https://huggingface.co/docs/datasets/en/cache

```bash
export HF_HOME=/data/.cache/huggingface
export HF_HUB_CACHE=/data/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data/.cache/huggingface/datasets
```

---

## 0.3 베이스라인 코드 템플릿(학습/추론/평가/배포 공통 뼈대)

### 레포 구조(권장)
```text
hf-course-baseline/
  README.md
  requirements.txt
  scripts/
    env.sh
    check_env.py
  src/
    train_seqclf.py
    eval_seqclf.py
    infer_seqclf.py
    serve_fastapi.py
```

### scripts/env.sh
```bash
#!/usr/bin/env bash
set -e
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[env] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
```

### scripts/check_env.py
```python
import torch, transformers, datasets, evaluate, huggingface_hub
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("evaluate:", evaluate.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
```

### 실행(최소 시나리오)
```bash
source scripts/env.sh
python scripts/check_env.py

python src/train_seqclf.py --epochs 1 --max_train_samples 2000 --max_eval_samples 500
python src/infer_seqclf.py --model_path outputs/seqclf --text "This is amazing!"
python src/eval_seqclf.py --model_path outputs/seqclf --max_samples 500

export MODEL_PATH=outputs/seqclf
uvicorn src.serve_fastapi:app --host 0.0.0.0 --port 8000
```

### 참고 링크
- Accelerate launch: https://huggingface.co/docs/accelerate/basic_tutorials/launch
- Text classification task guide: https://huggingface.co/docs/transformers/en/tasks/sequence_classification
