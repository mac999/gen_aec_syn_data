# 라우팅 재설계: 현황, 측정된 문제, 개선안

작성 2026-11-25. 기준 커밋 `6f1be32`. 측정은 v0.5.3 산출물(문서 1,539개,
RAG 청크 18,750개) 실측값이다. v0.5.4 재생성 전에 반영하는 것을 전제로 한다.

## 1. 현황

### 산출되는 코퍼스 8종 + VLM 자산 3종

문서 하나(`a.pdf`)가 들어가면 **문서명 폴더**가 생기고 그 안에 코퍼스별
파일이 놓인다. 파일명 자체가 종류를 나타내고, 레코드 id에 접두사가 붙는다.

| 코퍼스 | 파일 | id 접두사 | v053 산출 |
|---|---|---|---|
| DAPT | dapt_training_data.jsonl | `dapt_` | 742 |
| SFT | sllm_training_data.jsonl | `sft_` | 742 |
| SFT 탈락 | sllm_rejected.jsonl | 〃 | v054 신규 |
| STaR | star_training_data.jsonl | SFT id 승계 | 742 |
| STaR 탈락 | star_rejected.jsonl | 〃 | 742 |
| DPO | dpo_training_data.jsonl | `dpo_` | 690 |
| RLVR | rlvr_training_data.jsonl | `rlvr_` | 742 |
| RAG | rag_corpus.jsonl | `rag_` | 1,563 |
| VLM | vlm_training_data.jsonl | `vlm_` | 170 |

RAFT는 파일이 아니다. SFT 안의 `grounded_rag`(4,337행)·`refusal`(3,132행)이
쓰는 **기법**이며, `sft_tasks[].retrieval = "raft"`로 선택된다. RAG(코퍼스)와
RAFT(기법)는 층위가 다르다.

### 현재 라우팅

`src/doc_routing.py:classify()` 한 함수, **문서 단위 1회 판정**이다.

```
입력  파일명 + 첫 청크의 앞 2,000자        ← 문서 전체가 아님
처리  정규식 7개 가중합
      개정 +2.0 / 고시·훈령 +1.5 / 양식·서식 +1.5 / 제N호 +1.0 / 시행일 +0.5
      KDS·KCS·설계기준 -2.0 / 매뉴얼 -1.5
      (파일명 일치 = 전액, 본문 일치 = 절반)
출력  score >= 2.0 retrieve / 1.0~2.0 both / 미만 train
```

판정 근거 실측: 파일명 신호만 17.3%, 본문 신호만 40.7%, 둘 다 42.0%.

결정된 route가 `pipeline.py:181`에서 코퍼스 생성 여부로 **하드코딩** 연결된다.

| route | SFT | DAPT | RAG |
|---|---|---|---|
| train | O | O | X |
| both | O | O | O |
| retrieve | X | X | O |

## 2. 측정된 문제 5가지

1. **route 3값으로 코퍼스 8종을 제어할 수 없다.** "DAPT와 RAG만, SFT는 제외"
   같은 조합이 표현 불가. 매핑이 코드에 박혀 있어 설정으로 못 바꾼다.
2. **문서 단위 일괄 판정.** 한 문서 안에서 총칙은 안정적이고 별표 수치는
   자주 바뀌는데, 모든 청크가 같은 경로를 받는다.
3. **서식 문서가 RAG의 10.2%**(1,911청크). 그중 566청크는 수치가 용지규격
   (210mm×297mm)과 지질뿐이라 검색 답변 근거가 되지 못한다. `form` 신호가
   +1.5로 **의도적으로** 검색에 보내고 있다.
4. **provenance 결손.** revision_label 92.1% / document_no 72.6% /
   effective_date 68.8% / issuing_body 63.0% / **edition_year 0.0%**.
   edition_year 정규식이 `(2023)` 형태만 찾는데 실제는 `(제2023-98호)`다.
5. **개정이유서 과다.** 최다 발동 신호가 `amendment`(12,798회)인데 상당수가
   "◇ 제·개정 이유 … 일원화하고자 함" 류다. 무엇이 바뀌었는지는 알려주나
   현행 기준값에는 답하지 못한다.

부수적으로 `pipeline.py:181`의 retrieve 분기가 `want_star`·`want_rlvr`을
명시적으로 끄지 않는다. 지금은 `and sft_dicts` 가드 덕에 누수가 0이지만
(실측: retrieve 1,028문서 중 학습 산출물 0), SFT를 거치지 않는 경로가 생기면
바로 깨진다.

## 3. 규칙 기반의 한계

언어 무관 구조 지표로 서식과 조문을 가를 수 있는지 측정했다(문서당 첫 청크,
서식 784 vs 일반 766).

| 지표 | 서식 | 일반 | 차이 |
|---|---|---|---|
| 숫자 비율 | 0.061 | 0.070 | -0.008 |
| 평균 토큰 길이 | 3.58 | 3.76 | -0.18 |
| 구두점 비율 | 0.074 | 0.058 | +0.017 |
| 고유 토큰 비율 | 0.812 | 0.769 | +0.043 |
| **문장 종결 수** | **9.5** | **16.8** | **-7.3** |

문장 밀도 하나만 변별력이 있다. 정규식을 늘려도 영문·외국 규정이 들어오면
`고시|훈령|예규`를 언어마다 다시 써야 하므로 한계가 분명하다.

## 4. 측정된 해법: 임베딩 분류

로컬에 이미 있는 `nomic-embed-text`(0.3GB)로 서식 vs 조문 분류를 시험했다.

| 구성 | 정확도 |
|---|---|
| 손으로 쓴 프로토타입 2개, 접두사 없음 | **53%** (무작위) |
| `classification: ` 접두사 + 코퍼스에서 뽑은 예시 8+8 | **97%** |

두 가지가 결정적이다. nomic 계열은 **태스크 접두사가 필수**이고, 프로토타입은
**실제 코퍼스에서 뽑아야** 한다. 이 둘을 빼면 무작위로 떨어진다.

비용 실측: 건당 27ms. 문서 1,958개는 8병렬로 0.1분, 청크 31,814개는 1.8분.
현재 생성(qwen3:30b-a3b, 43시간) 대비 **0.5%**.

외부 조사와도 일치한다. COLM 2026 보고 기준 LLM 분류는 임베딩보다 최대
1,431배 비싸고, 분류 과제에서는 임베딩이 오히려 근소 우위다(0.62 vs 0.67).

## 5. 개선안: 4단 캐스케이드 + 선언형 라우트

```
L0  경로/파일명 글로브 → 유즈케이스 배정          비용 0, 사용자 선언
L1  구조 품질 게이트 (언어 무관)                  비용 0
      중복 줄 비율, 문장 밀도, 기호 비율, 여백
      미달 청크는 어느 코퍼스에도 넣지 않음
L2  임베딩 분류 (문서 + 청크)                     생성 비용의 0.5%
      코퍼스에서 뽑은 예시로 중심 구성 → 라벨 + 마진
L3  LLM 판정 — L2 마진이 낮은 것만                전체의 5~10%
      결과를 L2 예시 풀에 환류
```

### 5.1 `corpus_routes` — 선언형 매핑

`sft_tasks`·`vlm_tasks`와 같은 패턴으로 config에 둔다. 위에서부터 첫 매치를
쓰고, 마지막 항목이 기본값 역할을 한다.

```json
"corpus_routes": [
  { "name": "volatile_regulation",
    "when": { "embed_label": "amendment_notice", "min_margin": 0.03 },
    "emit": ["rag"] },
  { "name": "stable_standard",
    "when": { "glob": "01_design_standards_kds/**" },
    "emit": ["dapt", "sft", "star", "dpo", "rlvr"] },
  { "name": "form_template",
    "when": { "embed_label": "form", "min_margin": 0.03 },
    "emit": [] },
  { "name": "default", "when": {},
    "emit": ["dapt", "sft", "star", "rag"] }
]
```

- `emit`이 곧 게이트다. 코퍼스 8종을 개별 제어하므로 route 3값 제약이 사라진다.
- `when`에 `glob`(L0) / `regex`(현행 유지) / `embed_label`(L2) / `llm`(L3)을
  섞어 쓴다. 판정 방식이 교체 가능해진다.
- 유즈케이스 배정("이 파일명은 uc1")이 `glob` + `name`으로 표현된다.

### 5.2 `routing_classifier` — 분류기도 설정으로

품질의 핵심 변수가 **라벨 예시 목록**이므로(53% vs 97%) 반드시 외부화한다.
영문 코퍼스를 넣을 때 사용자가 바꿀 것도 정확히 이 부분이다.

```json
"routing_classifier": {
  "backend": "embedding",
  "model": "nomic-embed-text",
  "prefix": "classification: ",
  "labels": { "form": ["<코퍼스 예시 8개>"],
              "clause": ["<코퍼스 예시 8개>"],
              "amendment_notice": ["<코퍼스 예시 8개>"] },
  "uncertain_margin": 0.03,
  "fallback": "llm",
  "llm_prompt_template": "<프롬프트, sft_tasks와 같은 방식>"
}
```

### 5.3 청크 단위 전환

L1·L2를 청크에 적용하면 같은 문서에서 조문은 train, 별표는 retrieve로
갈라진다. 비용은 1.8분이다.

## 6. 작업 순서

| 단계 | 작업 | 규모 | 효과 |
|---|---|---|---|
| 0 | retrieve 분기에서 want_star·want_rlvr 명시적 차단, edition_year 정규식 수정 | 10분 | 취약점 제거 |
| 1 | L0 글로브 라우팅 + `corpus_routes` 선언 구조 | 1일 | 유즈케이스 배정이 선언형으로 |
| 2 | L1 구조 게이트(청크 단위) | 반나절 | 서식·여백 노이즈 제거, 언어 무관 |
| 3 | L2 임베딩 분류 + `routing_classifier` | 1일 | 가성비 최고. 97%, 생성 비용의 0.5% |
| 4 | L3 LLM 캐스케이드 (마진 하위만) | 반나절 | 애매한 5~10%만 |
| 5 | fastText 증류 | 나중에 | 규모 확장 시 거의 무료 |

0~3단계로 §2의 문제 1·2·3·4가 해소된다. 각 단계마다 v053 코퍼스로 전후
비교를 내어 효과를 수치로 확인한다.

## 7. 호환성 주의

- `doc_routing` 설정값(train/retrieve/both/auto)은 유지한다. `corpus_routes`가
  없으면 기존 동작 그대로 — v053 재현이 깨지지 않아야 한다.
- 기존 `routing_signals`는 `when.regex`로 계속 쓸 수 있게 한다.
- 청크 단위 전환은 `rag_corpus.jsonl`의 행 구성을 바꾸므로, kcbench가 RAG
  코퍼스를 소비한다면 사전 공유가 필요하다.
