# Part 8. Capstone (통합 과제)

## 8.1 미션(권장)
- “이미지 + 문서 기반 제품 상담”
  - 입력: (선택) 제품 이미지 + 사용자 질문
  - 기능: 멀티모달 이해(VLM) + 문서 검색(RAG) + 툴 호출(Agent)
  - 출력: 답변 + 근거(문서/섹션) + 필요한 경우 추가 질문

---

## 8.2 요구사항(필수)
- 근거 기반 답변
  - 근거 없으면 “모르겠습니다/추가 정보 필요”로 처리
- 실패 대응
  - 검색 결과 없음 → query rewrite 후 1회 재검색
  - 여전히 없음 → clarifying question 1개만 던지고 종료
- 성능/비용
  - p95 latency 목표 설정(예: 3초)
  - 1요청당 최대 토큰/최대 툴 호출 횟수 제한

---

## 8.3 아키텍처(권장 블록)
- Ingest/Index
  - 문서 → chunk → embedding → FAISS(또는 VectorDB)
- Runtime
  - Router(질문 분류: RAG 필요/불필요)
  - Retriever + Reranker
  - Generator(LLM/SLLM)
  - Agent(tool): spec lookup, 계산, 정책 체크 등
- Observability
  - request id, tool log, retrieval hits, cost estimate

---

## 8.4 단계별 마일스톤(제출 단위)
- M1: RAG only
  - 질문 20개에 대해 근거 포함 답변
- M2: Rerank 추가 + 평가 리포트
  - Retriever only vs +Rerank 비교
- M3: Agent 루프 추가
  - 재검색/추가질문/툴 호출 포함
- M4: 데모 배포
  - FastAPI 또는 Spaces(Gradio)

---

## 8.5 테스트 시나리오(최소 12개)
- 검색 성공 4개(정답 포함)
- 검색 실패 2개(문서에 없음)
- 모호 질문 2개(추가 질문 유도)
- 수치 계산/비교 2개(툴 사용 유도)
- 이미지 기반 질의 2개(VLM 사용)

---

## 8.6 제출물(필수)
- 코드 레포(실행 커맨드 포함)
- 인덱스 생성 스크립트 + 버전
- 평가 리포트(md/pdf)
- 데모 URL 또는 실행 방법
- 운영 체크리스트(Part 7 템플릿 기반)

---

## 8.7 평가(예시)
- 기능 완성도 35%
- 근거/정확성 25%
- 재현성/엔지니어링 20%
- 성능/비용 관리 10%
- 데모/사용성 10%
