# Part 7. 프로덕션 체크리스트 (옵션/심화)

## 7.1 관측가능성(필수)
- 로그 레벨
  - INFO: 요청 요약, 모델 버전, latency
  - DEBUG: tool 입력/출력(PII 마스킹)
- 트레이싱
  - 요청 ID로 end-to-end 추적
- 프롬프트/설정 버저닝
  - prompt template 파일화 + git tag

---

## 7.2 비용 최적화(현업 핵심)
- 캐시
  - retrieval 캐시, generation 캐시(동일 질의)
- 토큰 절감
  - 컨텍스트 상한, top-k 튜닝, chunk 길이 조정
- 배치/스트리밍
  - 서빙 레이어에서 batch 가능 여부 확인

---

## 7.3 성능 최적화(기본)
- 모델
  - FP16 / int8 / int4(품질과 함께 검증)
- 시스템
  - pinned memory, num_workers, I/O 병목 제거
- RAG
  - 인덱스 타입/파라미터 튜닝, rerank top-k 최소화

---

## 7.4 보안/컴플라이언스
- 데이터
  - PII 제거/마스킹, 접근 권한 분리
- 라이선스
  - 모델/데이터 license 확인(상업적 사용 가능 여부)
- 프롬프트 주입 방어
  - system prompt 보호, tool 호출 제한, 출력 검증

---

## 7.5 운영 문서(런북) 템플릿(최소)
- 서비스 목적/범위
- 의존성(모델 버전/인덱스 버전/DB)
- SLO
  - p95 latency, error rate
- 장애 대응
  - fallback 모델/기능 축소 모드
- 배포 절차
  - smoke test / rollback

---

## 7.6 결과물 체크리스트
- (필수) SLO 2개 정의 + 측정 방법
- (필수) 장애 시나리오 3개(runbook)
- (권장) 비용 추정(일 요청량 기준) + 상한 설계
