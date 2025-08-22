// static/js/mci_js/mci_input.js
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#predictForm");
  if (!form) return;

  // === 요소 캐싱 ===
  const age = document.getElementById("age");
  const edu = document.getElementById("edu_level");
  const period = document.getElementById("base_yrs");
  const genderRadios = document.querySelectorAll("input[name='gender']");
  const dbRadios = document.querySelectorAll("input[name='has_db']");
  const hibpeRadios = document.querySelectorAll("input[name='has_hibpe']");
  const cogRadios = document.querySelectorAll("input[name='cog_input_mode']");  // "mci" | "mmse"
  const mciRadios = document.querySelectorAll("input[name='has_mci_ui']");
  const mmseInput = document.getElementById("mmse_score");

  const mciField = document.getElementById("mci_input_field");
  const mmseField = document.getElementById("mmse_input_field");

  const hasMciField = document.getElementById("has_mci_hidden"); // 서버로 제출되는 히든 필드

  // 섹션이 없으면 더 진행하지 않음(토글 에러 방지)
  if (!mciField || !mmseField) return;

  // === 유틸: 라디오 선택값 ===
  const checkedVal = nodes => {
    for (const el of nodes) if (el.checked) return el.value;
    return null;
  };

  // === 인지 입력 필드 토글(+무관 입력 비활성화, 히든 값 동기화) ===
  function toggleCognitive() {
    const mode = checkedVal(cogRadios) || "mci"; // 기본 mci

    // 보이기/숨기기
    mciField.classList.toggle("hidden", mode !== "mci");
    mmseField.classList.toggle("hidden", mode !== "mmse");

    // 무관 입력 비활성화(의도치 않은 값 제출 방지)
    mciRadios.forEach(r => (r.disabled = mode !== "mci"));
    if (mmseInput) mmseInput.disabled = (mode !== "mmse");

    // 히든 값 초기화/동기화
    if (mode === "mci") {
      const v = checkedVal(mciRadios);
      hasMciField.value = v == null ? "" : v;    // mci 모드에서는 라디오 선택값 반영
    } else {
      hasMciField.value = "";                    // mmse 모드에서는 제출 직전에 계산해서 채움
    }
  }

  // MMSE/MCI 라디오가 바뀔 때마다 토글 실행 (브라우저별 이벤트 보강)
  ["change", "input", "click"].forEach(evt =>
    cogRadios.forEach(r => r.addEventListener(evt, toggleCognitive))
  );
  // 초기 1회
  toggleCognitive();

  // MCI 라디오 변경 시 히든 필드 동기화 (mci 모드에서만)
  mciRadios.forEach(r => {
    r.addEventListener("change", () => {
      if (checkedVal(cogRadios) === "mci") {
        const v = checkedVal(mciRadios);
        hasMciField.value = v == null ? "" : v;
      }
    });
  });

  // === 폼 제출 처리 ===
  form.addEventListener("submit", e => {
    e.preventDefault(); // 검증 후 수동 제출

    // 1) 공통 필수값 먼저 검사 (비어있으면 즉시 반환)
    if (!age.value.trim()) { alert("나이를 입력해 주세요."); age.focus(); return; }
    const gender = checkedVal(genderRadios);
    if (gender == null) { alert("성별을 선택해 주세요."); return; }
    if (!edu.value) { alert("교육 수준을 선택해 주세요."); edu.focus(); return; }

    const hasDb = checkedVal(dbRadios);
    if (hasDb == null) { alert("당뇨 여부를 선택해 주세요."); return; }

    const hasHibpe = checkedVal(hibpeRadios);
    if (hasHibpe == null) { alert("고혈압 여부를 선택해 주세요."); return; }

    if (!period.value.trim()) { alert("예측 기준 기간(년)을 입력해 주세요."); period.focus(); return; }

    // 간단한 범위 체크(즉시 피드백)
    const ageNum = parseInt(age.value, 10);
    if (Number.isNaN(ageNum) || ageNum < 65 || ageNum > 120) {
      alert("나이는 65~120 사이의 값으로 입력해 주세요."); age.focus(); return;
    }
    const yrsNum = parseInt(period.value, 10);
    if (Number.isNaN(yrsNum) || yrsNum < 1 || yrsNum > 10) {
      alert("예측 기준 기간(년)은 1~10 사이의 값으로 입력해 주세요."); period.focus(); return;
    }

    // 2) 인지 입력 분기 검사 (MCI/MMSE)
    const mode = checkedVal(cogRadios) || "mci";

    if (mode === "mci") {
      const mciVal = checkedVal(mciRadios);
      if (mciVal == null) { alert("MCI 여부를 선택해 주세요."); return; }
      hasMciField.value = String(mciVal);
    } else if (mode === "mmse") {
      const raw = (mmseInput?.value || "").trim();
      if (!raw) { alert("MMSE 점수를 입력해 주세요."); mmseInput?.focus(); return; }
      const mmseVal = parseInt(raw, 10);
      if (Number.isNaN(mmseVal)) { alert("MMSE 점수는 숫자여야 합니다."); mmseInput?.focus(); return; }
      if (mmseVal < 0 || mmseVal > 30) { alert("MMSE 점수는 0~30 사이로 입력해 주세요."); mmseInput?.focus(); return; }
      if (mmseVal <= 19) { alert("MMSE 점수가 19점 이하인 경우, 이미 치매로 의심되어 입력이 제한됩니다."); mmseInput?.focus(); return; }

      // 예시 변환 규칙: 20~26 → MCI(1), 27~30 → 정상(0)
      const derivedHasMci = (mmseVal <= 26) ? 1 : 0;
      hasMciField.value = String(derivedHasMci);
    }

    // 3) 결과 페이지에서 사용할 경우에만 저장
    if (period && period.value) {
      localStorage.setItem("base_yrs", String(period.value));
    }

    // 4) 모든 검증 통과 → 실제 제출
    form.submit();
  });
});
