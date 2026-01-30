# Part 4. SLLM(Small LLM) 엔지니어링: “작게, 싸게, 빠르게”

## 4.1 SLLM 전략(선택 기준)
- 쓰는 이유
  - 비용↓ / 지연시간↓ / 온프레미스·엣지 가능 / 데이터 유출 리스크↓
- 현실적 목표
  - “대형 모델 대체”가 아니라 “업무의 70% 커버 + 운영 가능”이 목적

---

## 4.2 QLoRA(기본 레시피)
- 라이브러리
  - transformers + peft + bitsandbytes (+ trl 사용 시 SFT 편함)
- 참고 링크
  - PEFT LoRA: https://huggingface.co/docs/peft/en/conceptual_guides/lora
  - bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes
  - TRL(SFT): https://huggingface.co/docs/trl

### 설치
```bash
pip install -U transformers accelerate peft bitsandbytes datasets trl
```

### QLoRA 로딩 스니펫(핵심)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "<small-llm-id>"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
```

### LoRA 어댑터(핵심)
```python
from peft import LoraConfig, get_peft_model

lora = LoraConfig(
  r=16, lora_alpha=32, lora_dropout=0.05,
  target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 모델 구조에 맞게 조정
  task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
```

---

## 4.3 (Lab A) SFT(Instruction 튜닝) 최소 예시(TRL)
- 데이터 포맷(권장)
  - `{"text": "### Instruction...\n### Response..."}`

### 스니펫(형태)
```python
from trl import SFTTrainer
from transformers import TrainingArguments

args = TrainingArguments(
  output_dir="outputs/sllm-sft",
  per_device_train_batch_size=2,
  gradient_accumulation_steps=8,
  num_train_epochs=1,
  learning_rate=2e-4,
  fp16=True,
  logging_steps=10,
  save_steps=200,
  report_to="none",
)

trainer = SFTTrainer(
  model=model,
  tokenizer=tok,
  train_dataset=train_ds,
  dataset_text_field="text",
  args=args,
  max_seq_length=1024,
)
trainer.train()
```

- 참고: TRL SFT 가이드: https://huggingface.co/docs/trl/en/sft_trainer

---

## 4.4 (Lab B) 속도/비용 벤치마크(필수)
- 측정 항목(최소)
  - p50/p95 latency
  - tokens/sec
  - VRAM 사용량
- 규칙
  - 동일 프롬프트/동일 max_new_tokens로 비교
- 스니펫(토큰/sec)
```python
import time, torch
from transformers import TextGenerationPipeline

pipe = TextGenerationPipeline(model=model, tokenizer=tok, device_map="auto")
prompt = "Write a short email about meeting schedule."
t0 = time.time()
out = pipe(prompt, max_new_tokens=128, do_sample=False)
dt = time.time() - t0
print("sec:", dt, "text:", out[0]["generated_text"][:120])
```

---

## 4.5 서빙(TGI/Endpoints) 개요
- 로컬/서버 서빙 표준: Text Generation Inference(TGI)
  - https://github.com/huggingface/text-generation-inference
- 운영 관점 체크
  - 배치/스트리밍 지원
  - 메모리/동시성
  - 토큰 제한/타임아웃

---

## 4.6 결과물 체크리스트
- (필수) QLoRA 학습 로그(지표/학습시간/VRAM)
- (필수) 벤치마크 표(원본 모델 vs 튜닝/양자화)
- (권장) “실패 프롬프트” 10개 모음 + 개선 전략(데이터/프롬프트/툴)
