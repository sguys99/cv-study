# Part 2. Vision AI 실전 (분류/검출/세그먼트)

## 2.1 모델 선택 가이드(한 줄)
- 분류: ViT / ConvNeXt
- 검출: DETR 계열
- 세그먼트: SegFormer 계열
- 공통: “전이학습 + 작은 데이터”에 강한 백본부터 시작

- 참고 링크
  - Vision task docs(Transformers): https://huggingface.co/docs/transformers/en/tasks/vision
  - Image classification: https://huggingface.co/docs/transformers/en/tasks/image_classification

---

## 2.2 데이터 파이프라인(실무 포인트)
- 이미지 증강(과하면 성능 저하)
  - baseline: resize + normalize + (optional) random crop/flip
- 라벨 포맷
  - 분류: class id
  - 검출: COCO(바운딩박스 + category)
  - 세그: mask + label map
- Datasets로 관리
  - `load_dataset(...)` → `map(preprocess)` → `train_test_split`

---

## 2.3 (Lab A) ViT 분류 fine-tune (Trainer)

### 설치
```bash
pip install -U transformers datasets evaluate accelerate pillow
```

### 코드(핵심 스니펫)
```python
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer

ds = load_dataset("beans")  # 예시(가벼운 비전 분류)
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess(examples):
    imgs = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(imgs, return_tensors="pt")
    examples["pixel_values"] = inputs["pixel_values"]
    return examples

ds = ds.with_transform(preprocess)

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,
)

args = TrainingArguments(
  output_dir="outputs/vit-beans",
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
```

- 참고 링크
  - beans dataset: https://huggingface.co/datasets/beans
  - ViT model hub: https://huggingface.co/google/vit-base-patch16-224

---

## 2.4 (Lab B) DETR 검출 파이프라인(핵심 흐름)
- 단계
  - COCO 포맷 데이터 준비 → image_processor로 전처리 → `AutoModelForObjectDetection`
- 참고(검출 태스크)
  - https://huggingface.co/docs/transformers/en/tasks/object_detection

### 추론 스니펫(간단)
```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

img = Image.open("sample.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
target_sizes = torch.tensor([img.size[::-1]])
results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
print(results["scores"][:5], results["labels"][:5], results["boxes"][:5])
```

---

## 2.5 (Lab C) SegFormer 세그먼트(핵심 흐름)
- 단계
  - mask 포함 데이터 → `AutoImageProcessor`(segmentation) → `AutoModelForSemanticSegmentation`
- 참고(세그 태스크)
  - https://huggingface.co/docs/transformers/en/tasks/semantic_segmentation

---

## 2.6 배포/최적화(필수 체크)
- ONNX 내보내기(옵션)
  - https://huggingface.co/docs/transformers/en/serialization#export-to-onnx
- 성능 체크
  - 지연시간(p50/p95), throughput, 배치 크기 변화
- 실전 팁
  - 먼저 FP16(가능한 GPU) → 이후 ONNX/TensorRT 고려
