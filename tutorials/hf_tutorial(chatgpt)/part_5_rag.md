# Part 5. RAG: 검색·재랭킹·평가까지 “끝장”

## 5.1 RAG 표준 파이프라인
- Ingest: 문서 수집 → 정제 → chunking
- Index: 임베딩 생성 → 벡터 인덱스 구축(FAISS 등)
- Retrieve: top-k 검색 + 필터링(metadata)
- Rerank: cross-encoder 등으로 top-k’ 재정렬
- Generate: 근거 컨텍스트 기반 답변 + 출처/근거 표시
- Evaluate: 정답 기반/비정답 기반 + 실패 유형 분류

---

## 5.2 (Lab A) 인덱싱: HF 임베딩 + FAISS

### 설치
```bash
pip install -U datasets faiss-cpu transformers sentence-transformers
```

### 인덱싱 스니펫(간단)
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embed_id = "sentence-transformers/all-MiniLM-L6-v2"
emb = SentenceTransformer(embed_id)

docs = ["doc text A ...", "doc text B ...", "doc text C ..."]
X = emb.encode(docs, normalize_embeddings=True)
X = np.array(X).astype("float32")

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

q = "question about B"
qv = emb.encode([q], normalize_embeddings=True).astype("float32")
scores, idx = index.search(qv, k=3)
print([(docs[i], float(scores[0][j])) for j, i in enumerate(idx[0])])
```

- 참고 링크
  - sentence-transformers: https://www.sbert.net/
  - FAISS: https://github.com/facebookresearch/faiss

---

## 5.3 Chunking 규칙(현업용 최소)
- 기본(권장 시작점)
  - chunk size 500~800 tokens
  - overlap 50~100 tokens
- 문서 유형별
  - FAQ/스펙: 섹션 단위 chunk(표/항목 단위)
  - 가이드/매뉴얼: heading 기반 분할 + 문단 합치기
- 메타데이터
  - source url/file, section heading, product/model 등 필터 키

---

## 5.4 (Lab B) Reranker 추가(품질 급상승 포인트)
- 선택지
  - Cross-encoder 계열(느리지만 품질↑)
- 형태
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(q, docs[i]) for i in idx[0]]
rerank_scores = reranker.predict(pairs)
order = np.argsort(-rerank_scores)
top = [idx[0][i] for i in order[:3]]
print([docs[i] for i in top])
```

---

## 5.5 (Lab C) Generate + 출처 표시
- 규칙
  - 답변에 사용한 문서 id/제목/구간을 함께 반환
- 형태(간단)
```python
from transformers import pipeline
gen = pipeline("text-generation", model="<llm>", device_map="auto")

context = "\n\n".join([docs[i] for i in top])
prompt = f"Answer using only the context.\n\nContext:\n{context}\n\nQ:{q}\nA:"
out = gen(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
print(out)
```

---

## 5.6 평가(RAGAS 등)
- 정답 기반
  - 정확도/루브릭/사람 평가
- 비정답 기반
  - faithfulness, context relevance, answer relevance 등
- 참고 링크
  - RAGAS: https://github.com/explodinggradients/ragas

---

## 5.7 결과물 체크리스트
- (필수) Retriever only vs +Rerank 품질 비교표
- (필수) 실패 유형 3종 분류(검색 실패/재랭킹 실패/생성 환각)
- (권장) “chunking 전략 A/B” 비교 실험
