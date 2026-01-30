# Part 6. Agent: 툴콜링부터 워크플로우 오케스트레이션까지

## 6.1 에이전트 최소 정의
- 입력: 사용자 목표(Goal)
- 내부: 상태(State) + 계획(Plan) + 실행(Act)
- 출력: 답변 + 사용한 툴 로그(가능하면)

---

## 6.2 Tool/function calling 설계 원칙
- 툴은 작게 쪼개기(단일 책임)
  - search_docs(query) / get_product_spec(model) / calculate(expr)
- 입력/출력 스키마 명확화(JSON)
- 실패 설계
  - timeout, retry, fallback, partial result

---

## 6.3 (Lab A) “3개 툴” 미니 에이전트(형태)
- 목표
  - 사용자의 질문을 분석 → 필요한 툴 호출 → 결과 합성

### 스켈레톤(간단)
```python
from typing import Dict, Any

def tool_search_docs(query: str) -> Dict[str, Any]:
    return {"hits": [{"id":"doc1","text":"..."}]}

def tool_calc(expr: str) -> Dict[str, Any]:
    import math
    return {"value": eval(expr, {"__builtins__": {}}, {"math": math})}

def tool_get_spec(model: str) -> Dict[str, Any]:
    return {"model": model, "spec": {"battery":"5000mAh"}}

TOOLS = {
  "search_docs": tool_search_docs,
  "calc": tool_calc,
  "get_spec": tool_get_spec,
}

def agent_step(plan: Dict[str, Any]) -> Dict[str, Any]:
    # plan 예: {"tool":"search_docs","args":{"query":"..."}}
    fn = TOOLS[plan["tool"]]
    return fn(**plan["args"])
```

- 연결 포인트
  - “plan 생성”은 LLM이 담당(프롬프트로 JSON plan 출력)
  - “실행”은 위 코드가 담당(안전한 스키마 검증 권장)

---

## 6.4 (Lab B) RAG + Agent 결합(권장 구조)
- 루프
  - 질문 이해 → 검색(최소 1회) → 근거로 답변 생성
  - 부족하면: 추가 질문(clarifying) 또는 재검색(rewrite query)
- 핵심 포인트
  - 에이전트는 “툴 호출”만 담당, 답변은 “근거 컨텍스트”로 생성
  - 근거 없는 답변 금지(가드레일)

---

## 6.5 워크플로우 오케스트레이션(상태 머신)
- 분기 예시
  - relevance 낮으면 → query rewrite → retrieve 재시도
  - tool error 발생 → fallback tool → 실패 안내
- 추천: LangGraph 같은 상태 그래프(외부 프레임워크) 사용 시 가독성↑

---

## 6.6 안정성/운영 체크
- 비용 상한
  - max calls, max tokens, max retries
- 관측가능성
  - tool call 로그(입력/출력/소요시간)
  - 실패 원인 코드화
- 보안
  - 툴 입력 검증(특히 eval/DB)

---

## 6.7 결과물 체크리스트
- (필수) tool log 포함한 실행 예시 10개
- (필수) 실패 시나리오 5개(네트워크 오류/빈 검색결과 등) 처리 결과
- (권장) 회귀 테스트(시나리오 JSON) 20개
