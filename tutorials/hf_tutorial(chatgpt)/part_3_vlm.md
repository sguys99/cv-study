# Part 3. VLM(Visual Language Model) 구축 & 튜닝

## 3.1 VLM 유형
- 임베딩형(검색/매칭)
  - CLIP 계열: 이미지↔텍스트 공동 임베딩
- 생성형(캡셔닝/QA)
  - 이미지 입력 + 텍스트 생성(질문응답/설명)

- 참고 링크
  - CLIP 모델들: https://huggingface.co/models?pipeline_tag=zero-shot-image-classification
  - Image-to-text: https://huggingface.co/docs/transformers/en/tasks/image_to_text

---

## 3.2 (Lab A) CLIP 기반 이미지 검색(미니 RAG 느낌)
- 목표
  - 이미지 임베딩 저장 → 텍스트 쿼리 임베딩 → 코사인 유사도 top-k

### 코드(핵심 스니펫)
```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
proc = CLIPProcessor.from_pretrained(model_id)

# 1) 이미지 인덱싱
image_paths = ["a.jpg", "b.jpg", "c.jpg"]
images = [Image.open(p).convert("RGB") for p in image_paths]
img_inputs = proc(images=images, return_tensors="pt", padding=True)
with torch.no_grad():
    img_emb = model.get_image_features(**img_inputs)
img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

# 2) 텍스트 쿼리
query = "a cat on sofa"
txt_inputs = proc(text=[query], return_tensors="pt", padding=True)
with torch.no_grad():
    txt_emb = model.get_text_features(**txt_inputs)
txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

scores = (txt_emb @ img_emb.T).squeeze(0).numpy()
topk = scores.argsort()[::-1][:3]
print([(image_paths[i], float(scores[i])) for i in topk])
```

---

## 3.3 (Lab B) 생성형 VLM 추론(VQA/설명)
- 목표
  - “이미지 + 프롬프트”로 답 생성
- 참고
  - Transformers pipeline / model별 예제는 모델 카드에서 확인하는 게 제일 빠름

### 실행 패턴(가이드)
- 모델 카드에서 “Usage” 섹션 확인 → 그대로 실행 → 프롬프트만 교체
- 주의
  - VRAM 부족: `torch_dtype=float16`, `device_map="auto"` 활용
  - 긴 응답: `max_new_tokens` 제한

### 스니펫(형태만)
```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_id = "<vlm-model-id>"   # 모델 카드 기반 선택
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

img = Image.open("sample.jpg").convert("RGB")
prompt = "Describe the image in one sentence."

inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

---

## 3.4 데이터(instruction format) 최소 규칙
- 샘플 단위(권장)
  - `{"image": ..., "text": "User: ...\nAssistant: ..."}`
- 품질 체크
  - 프롬프트-정답 “정합성” 우선(잘못된 라벨이 성능을 바로 망침)
  - 중복/유사 샘플 과다 시 과적합/환각 유도

---

## 3.5 튜닝(PEFT 방향)
- 원칙
  - 풀파인튜닝보다 LoRA/QLoRA로 시작
  - “데이터 품질”이 성능 대부분 결정
- 참고 링크
  - PEFT: https://huggingface.co/docs/peft
  - Transformers fine-tuning guide: https://huggingface.co/docs/transformers/en/training

---

## 3.6 결과물 체크리스트
- (필수) 이미지 10장에 대해 “프롬프트 3종” 결과 샘플 첨부
- (필수) 실패 케이스 5개(왜 실패했는지 가설 포함)
- (권장) Spaces 데모(업로드 이미지 + 질의)
