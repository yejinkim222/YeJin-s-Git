document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#predictForm");

  // === 요소 캐싱 ===
  const age = document.getElementById("age");
  const edu = document.getElementById("edu_level");
  const period = document.getElementById("base_yrs");
  const genderRadios = document.querySelectorAll("input[name='gender']");
  const dbRadios = document.querySelectorAll("input[name='has_db']");
  const hibpeRadios = document.querySelectorAll("input[name='has_hibpe']");
  const cogRadios = document.querySelectorAll("input[name='cog_input_mode']");
  const mciRadios = document.querySelectorAll("input[name='has_mci_ui']");
  const mmseInput = document.getElementById("mmse_score");

  const mciField = document.getElementById("mci_input_field");
  const mmseField = document.getElementById("mmse_input_field");

  const hasMciField = document.getElementById("has_mci_hidden")

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

  // MCI 라디오 변경 시 히든 필드 동기화
  mciRadios.forEach(r=>{
    r.addEventListener("change",()=>{
      const v = checkedVal(mciRadios);
      if (v!=null) hasMciField.value=v;
    })
  })

  // === 폼 제출 처리
  form.addEventListener("submit", e => {
    e.preventDefault();  // ✅ 반드시 제일 위에 와야 함

    const mode = checkedVal(cogRadios);
    if (mode === "mci" && !checkedVal(mciRadios)) {
      alert("MCI 여부를 선택해 주세요."); return;
    }
    if (mode === "mmse") {
      if (!mmseInput.value.trim()) {
        alert("MMSE 점수를 입력해 주세요."); mmseInput.focus(); return;
      }
      const mmseVal = parseInt(mmseInput.value, 10);
      if (!isNaN(mmseVal) && mmseVal <= 19) {
        alert("MMSE 점수가 19점 이하인 경우, 이미 치매로 의심되어 입력이 제한됩니다.");
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

    console.log("📨 서버로 보낼 원시 입력값:", formData);

    // === 입력값을 localStorage에 저장
    Object.entries(formData).forEach(([key, value]) => {
      localStorage.setItem(key, value);
    });

    // === 페이지 이동
    //window.location.href = "mci_output.html";
    form.submit()
  });
});
