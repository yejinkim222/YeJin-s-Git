document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#predictForm");

  // === 요소 캐싱 ===
  const age = document.getElementById("age");
  const edu = document.getElementById("education");
  const period = document.getElementById("period");
  const genderRadios = document.querySelectorAll("input[name='gender']");
  const dbRadios = document.querySelectorAll("input[name='db']");
  const hibpeRadios = document.querySelectorAll("input[name='hibpe']");
  const cogRadios = document.querySelectorAll("input[name='cog_input_mode']");
  const mciRadios = document.querySelectorAll("input[name='has_mci']");
  const mmseInput = document.getElementById("mmse_score");

  const mciField = document.getElementById("mci_input_field");
  const mmseField = document.getElementById("mmse_input_field");

  // === 유틸: 라디오 선택값
  const checkedVal = nodes => {
    for (const el of nodes) if (el.checked) return el.value;
    return null;
  };

  // === 인지 필드 토글
  function toggleCognitive() {
    const mode = checkedVal(cogRadios); // "mci" | "mmse"
    mciField.classList.toggle("hidden", mode !== "mci");
    mmseField.classList.toggle("hidden", mode !== "mmse");
  }
  cogRadios.forEach(r => r.addEventListener("change", toggleCognitive));
  toggleCognitive(); // 초기 실행 1회

  // === 폼 제출 처리
  form.addEventListener("submit", e => {
    e.preventDefault();

    // === 입력 검증
    if (!age.value.trim()) {
      alert("나이를 입력해 주세요."); age.focus(); return;
    }
    if (!checkedVal(genderRadios)) {
      alert("성별을 선택해 주세요."); return;
    }
    if (!edu.value) {
      alert("교육 수준을 선택해 주세요."); edu.focus(); return;
    }
    if (!checkedVal(dbRadios)) {
      alert("당뇨 여부를 선택해 주세요."); return;
    }
    if (!checkedVal(hibpeRadios)) {
      alert("고혈압 여부를 선택해 주세요."); return;
    }

    const mode = checkedVal(cogRadios);
    if (mode === "mci" && !checkedVal(mciRadios)) {
      alert("MCI 여부를 선택해 주세요."); return;
    }
    if (mode === "mmse") {
      if (!mmseInput.value.trim()) {
        alert("MMSE 점수를 입력해 주세요."); mmseInput.focus(); return;
      }
      const mmseVal = parseInt(mmseInput.value, 10);
      if (!isNaN(mmseVal) && mmseVal <= 17) {
        alert("MMSE 점수가 17점 이하인 경우, 이미 치매로 의심되어 입력이 제한됩니다.");
        mmseInput.focus(); return;
      }
    }

    if (!period.value.trim()) {
      alert("예측 기간을 입력해 주세요."); period.focus(); return;
    }

    // === 입력값 수집
    const formData = {
      age: age.value,
      education: edu.value,
      gender: checkedVal(genderRadios),
      db: checkedVal(dbRadios),
      hibpe: checkedVal(hibpeRadios),
      cog_input_mode: mode,
      has_mci: checkedVal(mciRadios),
      mmse_score: mmseInput.value,
      period: period.value
    };

    console.log("서버로 보낼 원시 입력값:", formData);

    // === 입력값을 localStorage에 저장
    Object.entries(formData).forEach(([key, value]) => {
      localStorage.setItem(key, value);
    });

    // === 페이지 이동
    window.location.href = "mci_output.html";
  });
});
