# v0.5.4 재생성 인수인계 (다른 PC 작업자용)

작성 2026-11-24. 생성기 저장소 gen_aec_syn_data, 기준 커밋 `462477f`.
`generator_v054_todo.md`(2026-11-24판)의 코드 수정 항목 1-1~1-8이 전부 구현·푸쉬된
상태에서 이어받는 문서다. kcbench 쪽(분할·학습·채점)과 병렬로 진행한다.

## 1. 무엇이 왜 바뀌었나

출발점은 kcbench worked-example-4의 발견이다: v0.5.3 STaR 행 12,725개 중
refusal이 2,346개(18.4%)였고(v0.5.2는 0.1%), 이 행들이 "제시된 자료만으로는
확인할 수 없습니다"를 기본값으로 가르쳐 학습된 모델이 지문이 있는 문항의
60%를 거절했다. 원인은 둘이다 — refusal 태스크가 "답 못 하는 질문을
지어내라"는 검증 불가능한 생성이었고, refusal 검증기가 거절 문구 정규식
하나라 거의 전부 통과시켰다.

`462477f`까지의 커밋 11개가 이것을 포함해 todo의 1-1~1-8을 고쳤다:

| 커밋 | 내용 |
|---|---|
| d0af2bd | refusal 검증기: 거절하면서 수치를 말하면 거부. 조항 인용은 "…조문이 필요합니다/참조" 같은 출처 요구 형태만 허용 |
| 1d0bc96 | refusal 쌍둥이: 검증된 grounded_rag QA의 질문을 재사용, 골든 뺀 distractor 블록을 context로, 답은 거절 템플릿. 원 답의 수치·인용이 distractor에 있으면 생성 안 함 → 답변 불가가 구성으로 보장됨 |
| decb4aa | star_task_share_cap: 검증 통과분에서 태스크 비중 상한(문서 단위, id 순 결정적). 초과분은 사유 달아 rejects로 |
| cafa8fd | holdout_file: kcbench holdout.json의 pdf_documents[*].generated_dir 문서는 SFT/STaR/DPO/RLVR 생성 제외(DAPT·rag 코퍼스는 유지). 파일이 설정됐는데 못 읽으면 실행 중단(오염 방지) |
| 6b2d9d5 | SFT 쓰기 전 numbers/citations 게이트(+sllm_rejected.jsonl), RLVR 문서 전체 폴백 제거(자기 청크 또는 스킵), DPO는 STaR 통과분만 + rejected 규칙(년·회 섭동 제외, 조사 잔여물 제외, 목차형 chosen 제외) |
| e62b69c 외 2 | 행 최상위 메타 doc_id/chunk_index/source_date/version/license, id는 `<doc_key>#<n>` (문서별 번호 — 재실행해도 다른 문서 id가 안 바뀜) |
| 349ef07 | max_samples_per_doc을 앞쪽 청크 소모가 아니라 전체 stride 분배로 (418청크 문서가 앞 4.3%만 쓰이던 문제) |
| 2f0b1a8, 0129800 | PDF 추출: 페이지 경계에서 끊긴 문장 봉합, 그림자 글자 겹침("2년"→"22년") 줄 밀도 판정으로 접기. 청크 길이는 700 유지(kcbench 키가 700자 기준) |
| 462477f | config_v054.json + 파일럿 게이트 scripts/check_v054_pilot.py |

### todo 문서와 다르게 한 결정 2건 (중요)

1. **refusal weight 0이 아니라 sft_tasks에서 제거.** select_task는 weight 0을
   걸러주지 않아 0으로 두면 오동작 여지가 있다. config_v054.json의 태스크는
   6종(regulation_qa, procedure, terminology, reasoning, structure,
   grounded_rag)이고 refusal 행은 쌍둥이가 공급한다(refusal_twin_ratio 0.3).
2. **검증기-쌍둥이 템플릿 충돌 조정.** todo 1-2대로 "조항 인용 전부 거부"로
   만들면 1-1의 쌍둥이 답("{doc}의 {section} 조문이 필요합니다")이 전멸한다.
   "단정 인용(제3조에 의하면 …이다) 거부, 출처 요구 인용(…조문이
   필요합니다/참조하십시오) 허용"으로 구현했다. kcbench 쪽에서 refusal 행을
   다시 채점할 때 이 규칙을 전제로 볼 것.

### 1-6 id 형식 변경 주의

행 id가 `sft_000001`에서 `<문서폴더명>#<n>`으로 바뀐다. kcbench 채굴·분할
코드가 id 형식을 가정한다면 v054 데이터를 읽기 전에 확인할 것. 최상위에
doc_id/chunk_index가 생겨 generated_dir 휴리스틱 파싱은 더 이상 필요 없다.

## 2. 실행 절차 (여유 GPU PC)

```bash
git pull                      # 462477f 이상
```

1. **kcbench holdout.json 배치.** config_v054.json의 holdout_file은
   `../kcbench/ground_truth_v053/holdout.json`이다. 이 파일은 생성기 리포에
   없고 kcbench 쪽 산출물이다. 경로가 다르면 config에서 고친다.
   파일이 없으면 실행이 즉시 중단된다 — 의도된 동작이다.
2. **파일럿: 문서 10개.** 입력 10개를 별도 폴더에 복사해 지정한다.
   ```bash
   python main.py -c config_v054.json --input-dir <10문서 폴더>
   python scripts/check_v054_pilot.py ../train_data_v054/train_data \
          ../kcbench/ground_truth_v053/holdout.json
   ```
   게이트 3검사: STaR 통과분의 refusal ≤ 8% / 쌍둥이 불변식(답에 수치 없음,
   evidence 빈 배열, context와 수치 교집합 없음) / 홀드아웃 문서의 감독 행 0.
   ALL CHECKS PASS가 아니면 전체 실행 금지.
   파일럿 표본에 홀드아웃 문서 1개와 대형 문서(KCS 시방서급) 1개를 섞으면
   필터와 stride 분배를 실전으로 확인할 수 있다.
3. **전체 재생성.** 약 43시간, qwen3:30b-a3b 기준 GPU 메모리 약 20GB.
   ```bash
   python main.py -c config_v054.json
   ```
   **v053 출력과 Google Drive의 v053은 절대 덮어쓰지 않는다** — output_dir이
   `../train_data_v054/train_data`로 분리돼 있으니 그대로 둘 것.
   새 게이트(6b2d9d5) 때문에 v053 대비 SFT 행 수가 줄고 sllm_rejected.jsonl이
   생긴다. 줄어든 것은 정상이다.
4. **전체 실행 후 게이트 재실행 + 결과 요약.** 문서 수, task_type별 행 수,
   STaR 통과율, sllm_rejected 행 수, 홀드아웃 제외 수(로그 말미에 찍힘),
   생성기 커밋 해시를 기록해 전달.

## 3. 이후 남은 일 (재생성 완료 후)

- **kcbench 쪽**: v054 데이터로 분할·학습·채점 재실행. 확인할 핵심은
  uc6 verdict 붕괴(v053에서 0.631→0.244)가 사라졌는지 — refusal 쌍둥이와
  8% cap이 이것을 겨냥한 수정이다. abstention율을 v052 수준(2~6%)과 비교.
- **생성기 쪽 미해결**:
  - 1-9 VLM 수정(render별 요소 ID·카메라 기록, 렌더→사진 복사 차단,
    하드 네거티브)은 이번에서 제외됨. 텍스트 트랙 무관. 다음 이미지 재생성 때.
  - 라이선스 원장(todo #10)·원천 확대(#17)는 사람 판정 대기.
- **Drive 업로드**: v054 완료분은 `1_syn_dataset/train_data_v054` 새 폴더로.
  rclone 절차는 dataset/UPLOAD.md 참조(--ignore-existing, sync 금지).

## 4. 빠른 검증 명령 모음

```bash
# 쌍둥이 행만 보기
grep -h '"task_type": "refusal"' <출력>/**/sllm_training_data.jsonl | head -3
# STaR 통과분 태스크 분포
python scripts/check_v054_pilot.py <출력>           # holdout 생략 시 1·2번만
# 홀드아웃 제외가 동작했는지 (실행 로그 말미)
grep "Holdout filter withheld" <실행로그>
```
