"use strict";

const kind = document.querySelector("#kind");
const editor = document.querySelector("#config");
const source = document.querySelector("#source");
const objective = document.querySelector("#objective");
const fields = document.querySelector("#method-fields");
const adviceResults = document.querySelector("#advice-results");

const defaults = {
  eda: {chart: "histogram", sample: 2500, bins: 20, correlation: "spearman"},
  model: {task: "regression", models: ["linear", "forest"], split: "random", test_size: 0.2,
    folds: 5, seed: 42, impute: "median", scale: "standard"},
  spc: {chart: "imr", run_rule: true},
  design: {design: "full", center: 3, repeat: 2, blocks: 1, seed: 42, budget: 30},
  doe: {},
  batch: {}
};

const forms = {
  eda: [
    ["chart", "차트", ["histogram", "frequency", "ecdf", "scatter", "trend"]],
    ["test", "통계 검정", ["", "pearson", "spearman", "welch", "mannwhitney", "welch_anova",
      "kruskal", "paired_t", "wilcoxon", "chisquare", "fisher"]],
    ["x", "X 열"], ["y", "Y 열"], ["group", "그룹 열"],
    ["alpha_test", "유의수준", "number"], ["alternative", "대립가설", ["", "two-sided", "less", "greater"]],
    ["correlation", "상관행렬", ["pearson", "spearman", "kendall"]],
    ["sample", "차트 표본 수", "number"], ["bins", "히스토그램 구간", "number"]
  ],
  model: [
    ["task", "과제 유형", ["regression", "classification"]], ["target", "목표 열"],
    ["features", "예측 특징 열 (줄바꿈)", "lines"], ["exclude", "제외 열 (줄바꿈)", "lines"],
    ["models", "후보 모델", ["linear", "tree", "forest", "boosting"], true],
    ["split", "검증 설계", ["random", "stratified", "group", "time", "group_time"]],
    ["group", "그룹 열"], ["time", "시간 열"], ["gap", "시간 분할 간격", "number"],
    ["folds", "교차검증 폴드", "number"], ["test_size", "홀드아웃 비율", "number"],
    ["scoring", "모델 선택 지표", ["", "neg_mean_absolute_error", "neg_root_mean_squared_error", "r2",
      "f1_macro", "balanced_accuracy", "roc_auc", "average_precision"]],
    ["impute", "수치형 결측 대치", ["median", "mean", "most_frequent"]],
    ["scale", "스케일링", ["standard", "robust", "none"]],
    ["clip", "학습 폴드 이상치 절단", ["", "iqr", "mad"]],
    ["clip_factor", "이상치 계수", "number"],
    ["power", "분포 변환", ["", "yeo-johnson", "box-cox"]],
    ["trees", "트리·반복 수", "number"], ["depth", "결정나무 최대 깊이", "number"],
    ["alpha", "Ridge 규제 강도", "number"], ["class_weight", "클래스 가중치", ["", "balanced"]],
    ["segments", "세그먼트 평가 열 (줄바꿈)", "lines"], ["seed", "무작위 시드", "number"]
  ],
  spc: [
    ["chart", "관리도", ["imr", "xbar_r", "xbar_s", "p", "u"]], ["value", "관측값 열"],
    ["time", "시간 열"], ["group", "부분군 열"], ["denominator", "분모 열"],
    ["start", "Phase I 시작", "date"], ["end", "Phase I 종료", "date"],
    ["lsl", "규격 하한", "number"], ["usl", "규격 상한", "number"], ["run_rule", "8점 연속 규칙", "checkbox"]
  ],
  design: [
    ["design", "실험 설계", ["full", "fractional", "ccd", "box_behnken"]],
    ["factors", "요인별 이름·저수준·고수준 (JSON 배열)", "json", false, '[{"name":"A","low":-1,"high":1},{"name":"B","low":-1,"high":1}]'],
    ["generators", "부분 요인 생성식 (JSON 객체)", "json", false, '{"C":["A","B"]}'],
    ["center", "중심점", "number"], ["repeat", "독립 반복", "number"], ["blocks", "블록", "number"],
    ["alpha", "축점 거리", "number"], ["budget", "실행 예산", "number"], ["seed", "무작위 시드", "number"]
  ],
  doe: [
    ["design_run", "승인된 설계 실행 ID"], ["response", "반응 열"],
    ["effects", "효과 모형", ["", "main", "interaction", "quadratic"]],
    ["terms", "선택한 추정 항 (줄바꿈)", "lines"],
    ["confirmation", "확인실험 설정 (JSON 객체)", "json", false, '{"baseline":[1,2,3],"candidate":[4,5,6],"minimum":1,"direction":"increase"}']
  ],
  batch: [["model_run", "승인된 모델 실행 ID"], ["id", "관측 식별자 열"],
    ["drift", "분포 변화 지표", ["", "psi", "js"]]]
};

const prepFields = [
  ["operation", "전처리 기법", ["numeric", "date", "unit", "range", "allowed", "drop_missing", "deduplicate",
    "ratio", "difference", "sum", "product", "log"]],
  ["column", "대상 열"], ["other", "두 번째 열"], ["output", "출력 열"],
  ["action", "위반 처리", ["", "flag", "mask", "exclude"]],
  ["min", "하한", "number"], ["max", "상한", "number"], ["values", "허용 범주 (JSON 배열)", "json"],
  ["factor", "단위 환산 계수", "number"], ["offset", "단위 환산 절편", "number"],
  ["keys", "중복 판정 키 (줄바꿈)", "lines"], ["format", "날짜 형식"]
];

const korean = {histogram:"히스토그램", frequency:"범주 빈도", ecdf:"누적분포", scatter:"산점도", trend:"추세",
  regression:"회귀", classification:"분류", median:"중앙값", mean:"평균", most_frequent:"최빈값",
  standard:"표준화", robust:"강건 스케일링", none:"미적용", append:"행 추가", left:"왼쪽 결합",
  inner:"교집합 결합", outer:"전체 결합", main:"주효과", interaction:"주효과 및 교호작용", quadratic:"2차 반응표면"};

function addFields(container, definitions, values, onChange, labels = {}) {
  definitions.forEach(([key, title, type, multiple, placeholder]) => {
    const label = document.createElement("label");
    label.textContent = title;
    const select = Array.isArray(type);
    const control = document.createElement(select ? "select" : ["lines", "json"].includes(type) ? "textarea" : "input");
    if (select) {
      control.multiple = Boolean(multiple);
      type.forEach(value => {
        const option = new Option(labels[value]?.[0] || korean[value] || value || "선택 안 함", value);
        option.selected = multiple ? (values[key] || []).includes(value) : value === (values[key] ?? type[0]);
        control.add(option);
      });
    } else if (type === "checkbox") {
      control.type = type;
      control.checked = Boolean(values[key]);
    } else {
      if (control.tagName === "INPUT") {
        control.type = type || "text";
        if (type === "number") control.step = "any";
        if (["x", "y", "group", "target", "time", "column", "other", "value", "denominator", "id", "response"].includes(key)) {
          control.setAttribute("list", "analysis-columns");
        }
      } else {
        control.rows = 2;
      }
      control.value = type === "lines" ? (values[key] || []).join("\n")
        : type === "json" && values[key] !== undefined ? JSON.stringify(values[key]) : values[key] ?? "";
      if (placeholder) control.placeholder = placeholder;
    }
    control.addEventListener("input", () => {
      try {
        const raw = control.value.trim();
        let value = raw;
        if (multiple) value = [...control.selectedOptions].map(option => option.value);
        else if (type === "checkbox") value = control.checked;
        else if (!raw) value = undefined;
        else if (type === "number") {
          value = Number(raw);
          if (!Number.isFinite(value)) throw new Error("유한한 숫자를 입력하세요.");
        } else if (type === "lines") value = raw.split("\n").map(item => item.trim()).filter(Boolean);
        else if (type === "json") value = JSON.parse(raw);
        onChange(key, value);
        control.setCustomValidity("");
      } catch (error) {
        control.setCustomValidity(error.message);
      }
    });
    label.append(control);
    container.append(label);
  });
}

function readConfig() {
  const config = JSON.parse(editor.value);
  if (!config || Array.isArray(config) || typeof config !== "object") throw new Error("설정은 JSON 객체여야 합니다.");
  return config;
}

function saveConfig(config) {
  editor.value = JSON.stringify(config, null, 2);
  document.querySelector("#config-error").textContent = "";
}

function renderFields() {
  const config = readConfig();
  const catalog = JSON.parse(document.querySelector("#method-catalog").textContent);
  fields.replaceChildren();
  const data = config.data ||= {};
  const options = config.options ||= {};
  const grid = document.createElement("div");
  grid.className = "approval";
  const labels = {...catalog[kind.value], ...catalog.validation};
  addFields(grid, forms[kind.value], options, (key, value) => {
    options[key] = value;
    saveConfig(config);
  }, labels);
  if (!["design", "doe", "batch"].includes(kind.value)) {
    const section = document.createElement("details");
    section.open = Boolean(data.rules?.length);
    const title = document.createElement("summary");
    title.textContent = "데이터 결합 및 전처리";
    section.append(title);
    const joinFields = document.createElement("div");
    joinFields.className = "approval";
    addFields(joinFields, [["combine", "다중 파일 처리", ["", "append"]],
      ["keys", "결합 키 (줄바꿈)", "lines"], ["join", "결합 방식", ["left", "inner", "outer"]],
      ["cardinality", "결합 관계", ["one_to_one", "one_to_many", "many_to_one", "many_to_many"]]],
    data, (key, value) => { data[key] = value; saveConfig(config); });
    section.append(joinFields);
    const rules = data.rules ||= [];
    rules.forEach((rule, index) => {
      const area = document.createElement("fieldset");
      const heading = document.createElement("legend");
      heading.textContent = "전처리 " + (index + 1);
      const ruleFields = document.createElement("div");
      ruleFields.className = "approval";
      addFields(ruleFields, prepFields, rule, (key, value) => {
        rule[key] = value;
        saveConfig(config);
      }, catalog.preprocessing);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "secondary compact";
      remove.textContent = "규칙 제거";
      remove.onclick = () => { rules.splice(index, 1); saveConfig(config); renderFields(); };
      area.append(heading, ruleFields, remove);
      section.append(area);
    });
    const add = document.createElement("button");
    add.type = "button";
    add.className = "secondary compact";
    add.textContent = "전처리 규칙 추가";
    add.onclick = () => {
      rules.push({operation: "numeric"});
      saveConfig(config);
      renderFields();
    };
    section.append(add);
    fields.append(section);
  }
  fields.append(grid);
}

function showExample() {
  saveConfig({options: defaults[kind.value]});
  renderFields();
}

function applyRecommendation(item) {
  const config = readConfig();
  const data = config.data ||= {};
  const options = config.options ||= {};
  if (item.stage === "preprocessing") {
    const rule = {...item.patch};
    for (const {role, column} of item.assignments) {
      if (["column", "other"].includes(role)) rule[role] = column;
    }
    (data.rules ||= []).push(rule);
  } else {
    if (item.stage === "eda" && !item.patch.test) delete options.test;
    Object.assign(options, item.patch);
    const features = [];
    for (const {role, column} of item.assignments) {
      if (role === "feature") features.push(column);
      else if (forms[kind.value].some(([key]) => key === role)) options[role] = column;
    }
    if (item.stage === "model" && features.length && !options.features?.length) {
      options.features = [...new Set(features)].filter(column => ![options.target, options.group, options.time].includes(column));
    }
  }
  saveConfig(config);
  renderFields();
}

async function postJSON(url, data) {
  const result = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
  const json = await result.json().catch(() => null);
  if (!result.ok || !json) throw new Error(json?.error || "요청을 완료하지 못했습니다.");
  return json;
}

if (kind && editor) {
  const reset = () => {
    const design = kind.value === "design";
    document.querySelector("#source-field").hidden = design;
    source.disabled = design;
    source.required = !design;
    adviceResults.replaceChildren();
    showExample();
  };
  reset();
  kind.addEventListener("change", reset);
  source.addEventListener("change", () => adviceResults.replaceChildren());
  document.querySelector("#load-example").onclick = showExample;
  editor.addEventListener("change", () => {
    try { renderFields(); }
    catch (error) { document.querySelector("#config-error").textContent = error.message; }
  });
  document.querySelector("#analysis-form").addEventListener("submit", event => {
    try {
      const config = readConfig();
      if (objective.value.trim()) (config.task_definition ||= {}).question = objective.value.trim();
      saveConfig(config);
    } catch (error) {
      event.preventDefault();
      document.querySelector("#config-error").textContent = error.message;
    }
  });
  const getAdvice = document.querySelector("#get-advice");
  getAdvice.onclick = async () => {
    const stage = kind.value;
    const selectedSources = () => source.disabled ? [] : [...source.selectedOptions].map(option => option.value);
    const sources = selectedSources();
    const purpose = objective.value;
    getAdvice.disabled = true;
    adviceResults.textContent = "자료 구조와 분석 목적을 검토 중입니다.";
    try {
      const result = await postJSON(getAdvice.dataset.url, {kind: stage, sources, objective: purpose, data: readConfig().data});
      if (kind.value !== stage || objective.value !== purpose ||
          JSON.stringify(sources) !== JSON.stringify(selectedSources())) return;
      adviceResults.replaceChildren();
      document.querySelector("#analysis-columns").replaceChildren(...result.columns.map(column => new Option(column)));
      const notes = document.createElement("p");
      notes.textContent = result.notes;
      adviceResults.append(notes);
      for (const item of result.recommendations) {
        const box = document.createElement("article");
        box.className = "advice-item";
        const heading = document.createElement("strong");
        heading.textContent = item.label;
        const reason = document.createElement("p");
        reason.textContent = item.reason;
        const assumptions = document.createElement("p");
        assumptions.textContent = item.assumptions;
        const choose = document.createElement("button");
        choose.type = "button";
        choose.className = "secondary compact";
        choose.textContent = "옵션에 적용";
        choose.onclick = () => {
          try {
            applyRecommendation(item);
            choose.textContent = "적용됨";
          } catch (error) {
            document.querySelector("#config-error").textContent = error.message;
          }
        };
        box.append(heading, reason, assumptions, choose);
        adviceResults.append(box);
      }
    } catch (error) {
      adviceResults.textContent = error.message;
    } finally {
      getAdvice.disabled = false;
    }
  };
}

const draft = document.querySelector("#draft-report");
if (draft) {
  draft.onclick = async () => {
    const status = document.querySelector("#draft-status");
    const form = document.querySelector("#approval-form");
    draft.disabled = true;
    status.textContent = "보고서 초안을 작성 중입니다.";
    try {
      const result = await postJSON(draft.dataset.url, {});
      for (const [name, value] of Object.entries(result)) {
        const field = form.elements.namedItem(name);
        if (field?.tagName === "TEXTAREA" && !field.value.trim()) field.value = value;
      }
      status.textContent = "빈 항목에 초안을 입력했습니다. 검토 후 승인하세요.";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      draft.disabled = false;
    }
  };
}
