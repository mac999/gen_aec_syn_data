/* Korean/English strings for the webview shell. */
(function () {
  const DICT = {
    ko: {
      "app.title": "합성 데이터셋 검토",
      "top.in": "입력", "top.out": "출력",
      "top.config": "config.json",
      "top.generate": "합성 데이터 생성",
      "top.stop": "중지",
      "top.exportIn": "입력 엑셀",
      "top.exportOut": "출력 엑셀",

      "left.title": "입력 파일",
      "left.search": "파일명 검색…",
      "right.title": "생성 결과",
      "right.search": "파일명 검색…",
      "dataset.title": "데이터셋 미리보기",
      "centre.empty": "파일을 선택하세요",

      "opt.title": "생성 옵션",
      "opt.apply": "적용",
      "opt.save": "config.json 저장",
      "opt.onlyNew": "신규만",
      "opt.dryRun": "드라이런",
      "opt.selectedOnly": "선택 파일만",

      "log.title": "실행 로그",
      "log.clear": "지우기",
      "log.idle": "대기 중",
      "log.running": "실행 중…",
      "log.done": "완료",
      "log.failed": "실패",
      "log.cancelled": "중지됨",

      "group.common": "공통", "group.llm": "LLM 백엔드", "group.sft": "SFT",
      "group.dapt": "DAPT", "group.pdf": "PDF 추출/청킹", "group.ifc": "IFC 렌더링",
      "group.vlm": "VLM", "group.comfyui": "ComfyUI 이미지 합성", "group.other": "기타",

      "msg.applied": "옵션이 적용되었습니다",
      "msg.saved": "config.json에 저장했습니다",
      "msg.noChange": "변경된 값이 없습니다",
      "msg.started": "생성을 시작했습니다",
      "msg.stopped": "중지 요청을 보냈습니다",
      "msg.noOutput": "이 입력 파일의 생성 결과가 없습니다",
      "msg.selectFile": "왼쪽/오른쪽 트리에서 파일을 선택하세요",
      "msg.loading": "불러오는 중…",
      "msg.no3d": "three.js를 불러오지 못했습니다 (인터넷 연결 필요)",
      "msg.noIfc": "IFC 지오메트리를 읽지 못했습니다",
      "msg.binary": "미리보기를 지원하지 않는 형식입니다",
      "msg.confirmRun": "합성 데이터 생성을 시작할까요?",

      "view.page": "페이지", "view.of": "/", "view.find": "본문 검색…",
      "view.hits": "건", "view.zoom": "배율", "view.open": "원본 열기",
      "view.elements": "요소", "view.reset": "시점 초기화",
      "view.records": "레코드", "view.prev": "이전", "view.next": "다음",
      "view.all": "전체 출력 보기",

      "stat.files": "파일", "stat.records": "레코드", "stat.images": "이미지"
    },

    en: {
      "app.title": "Synthetic Dataset Review",
      "top.in": "IN", "top.out": "OUT",
      "top.config": "config.json",
      "top.generate": "Generate",
      "top.stop": "Stop",
      "top.exportIn": "Export inputs",
      "top.exportOut": "Export outputs",

      "left.title": "Input files",
      "left.search": "Filter by name…",
      "right.title": "Generated output",
      "right.search": "Filter by name…",
      "dataset.title": "Dataset preview",
      "centre.empty": "Select a file",

      "opt.title": "Generation options",
      "opt.apply": "Apply",
      "opt.save": "Save config.json",
      "opt.onlyNew": "Only new",
      "opt.dryRun": "Dry run",
      "opt.selectedOnly": "Selected file only",

      "log.title": "Run log",
      "log.clear": "Clear",
      "log.idle": "Idle",
      "log.running": "Running…",
      "log.done": "Finished",
      "log.failed": "Failed",
      "log.cancelled": "Cancelled",

      "group.common": "Common", "group.llm": "LLM backend", "group.sft": "SFT",
      "group.dapt": "DAPT", "group.pdf": "PDF extraction", "group.ifc": "IFC rendering",
      "group.vlm": "VLM", "group.comfyui": "ComfyUI synthesis", "group.other": "Other",

      "msg.applied": "Options applied",
      "msg.saved": "Saved to config.json",
      "msg.noChange": "Nothing changed",
      "msg.started": "Generation started",
      "msg.stopped": "Stop requested",
      "msg.noOutput": "No generated output for this input file",
      "msg.selectFile": "Pick a file in the left or right tree",
      "msg.loading": "Loading…",
      "msg.no3d": "three.js could not be loaded (needs internet access)",
      "msg.noIfc": "Could not read IFC geometry",
      "msg.binary": "No preview for this file type",
      "msg.confirmRun": "Start synthetic data generation?",

      "view.page": "Page", "view.of": "/", "view.find": "Find in text…",
      "view.hits": "hits", "view.zoom": "Zoom", "view.open": "Open raw",
      "view.elements": "elements", "view.reset": "Reset view",
      "view.records": "records", "view.prev": "Prev", "view.next": "Next",
      "view.all": "Show whole output tree",

      "stat.files": "files", "stat.records": "records", "stat.images": "images"
    }
  };

  let lang = localStorage.getItem("aec.lang") || "ko";

  const I18N = {
    get lang() { return lang; },
    t(key) { return (DICT[lang] && DICT[lang][key]) || (DICT.ko[key]) || key; },
    set(next) {
      lang = DICT[next] ? next : "ko";
      localStorage.setItem("aec.lang", lang);
      document.documentElement.lang = lang;
      I18N.apply();
      window.dispatchEvent(new CustomEvent("aec:lang", { detail: lang }));
    },
    toggle() { I18N.set(lang === "ko" ? "en" : "ko"); },
    apply(root) {
      (root || document).querySelectorAll("[data-i18n]").forEach(el => {
        el.textContent = I18N.t(el.dataset.i18n);
      });
      (root || document).querySelectorAll("[data-i18n-ph]").forEach(el => {
        el.placeholder = I18N.t(el.dataset.i18nPh);
      });
    }
  };

  window.I18N = I18N;
})();
