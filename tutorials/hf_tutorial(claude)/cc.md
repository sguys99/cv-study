## 교육 대상

Vision AI, VLM, sLLM, RAG, Agent 엔지니어들을 대상으로 하니, 기본적인 ML/DL 지식은 있다고 가정하고 Hugging Face 생태계 활용에 집중하겠습니다.

---

## 교육 과정 목차 (안)

### Part 1: Hugging Face 생태계 기초
1. **Hugging Face Hub 이해하기**
   - Hub 구조: Models, Datasets, Spaces
   - Model Card 읽는 법, 좋은 모델 선별 기준
   - `huggingface_hub` 라이브러리 활용

2. **Transformers 라이브러리 핵심**
   - Pipeline API로 빠른 프로토타이핑
   - AutoClass 패턴 (AutoModel, AutoTokenizer, AutoProcessor)
   - from_pretrained / push_to_hub 워크플로우

---

### Part 2: Vision AI & VLM
3. **Vision Transformer 실습**
   - ViT, DeiT, Swin Transformer 비교
   - Image Classification, Object Detection (DETR, YOLOS)
   - `transformers`의 `AutoImageProcessor` 활용

4. **Vision-Language Models**
   - CLIP: 이미지-텍스트 임베딩 정렬
   - BLIP-2, LLaVA: Visual Question Answering
   - PaliGemma, Qwen-VL 등 최신 VLM 실습
   - Multimodal Pipeline 구축

5. **VLM Fine-tuning**
   - LoRA/QLoRA를 활용한 효율적 튜닝
   - Custom dataset 준비 (이미지-캡션 쌍)
   - `trl`의 `SFTTrainer` 활용

---

### Part 3: Small LLM (sLLM) 활용
6. **sLLM 선택과 배포**
   - Phi, Gemma, Qwen 등 sLLM 비교
   - Quantization: bitsandbytes, GPTQ, AWQ
   - `transformers`의 4-bit/8-bit 로딩

7. **sLLM Fine-tuning**
   - Instruction Tuning 데이터셋 구성
   - `trl` + `peft`로 LoRA 튜닝
   - RLHF/DPO 기초

8. **로컬 배포**
   - `text-generation-inference` (TGI) 서버
   - vLLM과의 비교
   - Gradio/Spaces로 데모 배포

---

### Part 4: RAG 시스템 구축
9. **Embedding & Vector Search**
   - Sentence Transformers 활용
   - `datasets`의 FAISS 인덱싱
   - 한국어 임베딩 모델 선택 가이드

10. **RAG Pipeline 구현**
    - Document Chunking 전략
    - Retriever + Generator 연결
    - `langchain` + Hugging Face 통합

11. **고급 RAG 패턴**
    - Reranking (Cross-encoder)
    - Hybrid Search (BM25 + Dense)
    - Self-RAG, Corrective RAG 개념

---

### Part 5: Agent 시스템
12. **Hugging Face Agents (smolagents)**
    - Tool 정의와 Agent 구조
    - CodeAgent vs ToolCallingAgent
    - Multi-step reasoning

13. **Custom Tool 개발**
    - Vision Tool, Search Tool 만들기
    - MCP (Model Context Protocol) 연동
    - Agent + RAG 결합

14. **실전 Agent 프로젝트**
    - 멀티모달 Agent (이미지 분석 + 웹 검색)
    - 자율 연구 Agent 구축
    - 평가와 디버깅

---

### Part 6: Production & Best Practices
15. **모델 최적화**
    - ONNX 변환, TensorRT
    - `optimum` 라이브러리
    - Inference Endpoints 활용

16. **MLOps on Hugging Face**
    - Spaces로 CI/CD
    - Model versioning
    - Evaluation with `evaluate` 라이브러리

---