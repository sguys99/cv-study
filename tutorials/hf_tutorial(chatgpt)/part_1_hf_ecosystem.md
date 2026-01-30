# Part 1. Hugging Face 생태계 “엔지니어링 관점” 빠른 정리

## 1.1 핵심 컴포넌트 맵
- Transformers: 모델/토크나이저/파이프라인(학습/추론 API)
  - https://huggingface.co/docs/transformers
- Datasets: 데이터 로딩/버전/캐시/전처리(map)
  - https://huggingface.co/docs/datasets
- Evaluate: 메트릭 표준화(accuracy, f1, bleu, rouge, mAP 등)
  - https://huggingface.co/docs/evaluate
- Accelerate: 멀티GPU/분산/혼합정밀(launch 중심)
  - https://huggingface.co/docs/accelerate
- Hub: 모델/데이터/Spaces 저장소, 버저닝, 권한/토큰
  - https://huggingface.co/docs/hub

---

## 1.2 Trainer vs Accelerate (선택 기준)
- Trainer(빠른 생산성)
  - 장점: 학습 루프/저장/평가/로그 표준 제공
  - 추천: “베이스라인 만들기”, 작은-중간 규모 튜닝
- Accelerate(유연성/성능)
  - 장점: 커스텀 루프 + 분산/혼합정밀을 깔끔히
  - 추천: 대규모/커스텀 손실/멀티모달/특수 배치 로직

- 판단 3줄 룰
  - “표준 태스크 + 빠른 결과” → Trainer
  - “커스텀 루프/모델/데이터 로더” → Accelerate
  - “둘 다” 가능하면: Trainer로 먼저 베이스라인 → 성능/유연성 필요 시 Accelerate로 이관

---

## 1.3 Hub 운영(버전/태깅/재현성)
- 모델 카드(Model Card) 필수 항목
  - 데이터 출처/라이선스
  - 학습 설정(모델/에폭/러닝레이트/배치)
  - 평가 지표
  - 사용 예시(추론 코드)
  - 한계/주의사항
  - 가이드: https://huggingface.co/docs/hub/model-cards

- 실험 버저닝 규칙(권장)
  - 모델 repo: `org/project-task-YYYYMMDD-expNN`
  - 태그: `v0.1-baseline`, `v0.2-qlora`, `v0.3-rerank`
  - README에 “실행 커맨드” 고정

---

## 1.4 (Lab) “실습 레포” 표준 구조 + Hub push
- 목표
  - (1) 로컬에서 학습/평가
  - (2) Hub로 푸시
  - (3) 다른 환경에서 pull 후 동일 동작

### 실행
```bash
# 환경
pip install -U transformers datasets evaluate accelerate huggingface_hub

# 로그인
hf auth login

# 학습 + 푸시(Part 0 템플릿 활용)
python src/train_seqclf.py --push_to_hub --hub_repo <username>/seqclf-baseline
```

### 코드(푸시 핵심만)
```python
training_args = TrainingArguments(
  ...,
  push_to_hub=True,
  hub_model_id="username/seqclf-baseline",
)
trainer.push_to_hub()
```

- 참고 링크
  - huggingface_hub quick start: https://huggingface.co/docs/huggingface_hub/en/quick-start
  - git credential 연동: https://huggingface.co/docs/huggingface_hub/en/guides/cli#authentication
