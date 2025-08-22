// static/js/llm_js/screen_audio_chat.js
document.addEventListener("DOMContentLoaded", () => {
  const logEl   = document.getElementById("screen-log");
  const recEl   = document.getElementById("recognized");
  const btnStart  = document.getElementById("btnStart");
  const btnRec    = document.getElementById("btnRec");
  const btnExtend = document.getElementById("btnExtend");
  const btnNext   = document.getElementById("btnNext");
  const timeLeft  = document.getElementById("timeLeft");

  // --- 간단한 상태(타이머) ---
  let secondsLeft = 60;
  let timerId = null;

  function appendLine(who, text) {
    logEl.textContent += `\n${who}: ${text}`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function startTimer() {
    clearInterval(timerId);
    timerId = setInterval(() => {
      secondsLeft = Math.max(0, secondsLeft - 1);
      timeLeft.textContent = `남은 시간 ${secondsLeft}s`;
      if (secondsLeft === 0) {
        clearInterval(timerId);
      }
    }, 1000);
  }

  function resetTimer(sec = 60) {
    secondsLeft = sec;
    timeLeft.textContent = `남은 시간 ${secondsLeft}s`;
    startTimer();
  }

  // --- 버튼 동작(임시 목업) ---
  btnStart.addEventListener("click", () => {
    appendLine("SolCare", "오늘 날짜를 말해 주세요. (예: 이천이십오년 팔월 이십일 또는 2025-08-21)");
    resetTimer(60);
  });

  btnRec.addEventListener("click", () => {
    // 추후: MediaRecorder 시작/정지 + 서버 전송 + Whisper 결과를 recEl.value에 채우기
    appendLine("나", "(음성 인식 결과가 여기에 추가됩니다)");
    recEl.focus();
  });

  btnExtend.addEventListener("click", () => {
    secondsLeft += 30;
    timeLeft.textContent = `남은 시간 ${secondsLeft}s`;
  });

  btnNext.addEventListener("click", () => {
    const userText = (recEl.value || "").trim();
    if (userText) {
      appendLine("나", userText);
      recEl.value = "";
    }
    // 추후: 서버로 답변 제출 → 다음 질문 받아서 appendLine("SolCare", nextQuestion)
    appendLine("SolCare", "다음 질문을 준비 중입니다…");
    resetTimer(60);
  });
});
