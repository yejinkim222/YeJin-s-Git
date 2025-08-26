// static/js/genai_js/genai_audio.js
document.addEventListener("DOMContentLoaded", () => {
  const API = window.API_GENAI_AUDIO_CONVERSE || "/genai/audio/converse";

  const chat     = document.getElementById("chat");
  const chatList = document.getElementById("chatList");
  const recEl    = document.getElementById("recognized");
  const btnStart = document.getElementById("btnStart");
  const btnRec   = document.getElementById("btnRec");
  const btnSend  = document.getElementById("btnSend");
  const btnExt   = document.getElementById("btnExtend");
  const timeEl   = document.getElementById("timeLeft");

  // 타이머
  let secondsLeft = 60, timerId = null;
  function startTimer() {
    clearInterval(timerId);
    timerId = setInterval(() => {
      secondsLeft = Math.max(0, secondsLeft - 1);
      timeEl.textContent = `남은 시간 ${secondsLeft}s`;
      if (secondsLeft === 0) clearInterval(timerId);
    }, 1000);
  }
  function resetTimer(sec = 60) { secondsLeft = sec; timeEl.textContent = `남은 시간 ${secondsLeft}s`; startTimer(); }

  // 말풍선
  function addMsg(role, text) {
    if (!text) return;
    const li = document.createElement("li");
    li.className = "msg " + (role === "user" ? "user" : "bot");
    const bubble = document.createElement("div");
    bubble.className = "bubble " + (role === "user" ? "user" : "bot");
    bubble.textContent = text;
    li.appendChild(bubble);
    chatList.appendChild(li);
    chat.scrollTop = chat.scrollHeight;
  }
  function addTyping() {
    const li = document.createElement("li");
    li.className = "msg bot";
    const b = document.createElement("div");
    b.className = "bubble bot"; b.textContent = "입력 중…";
    li.appendChild(b); chatList.appendChild(li); chat.scrollTop = chat.scrollHeight;
    return li;
  }
  function removeTyping(node) { if (node && node.parentNode) node.parentNode.removeChild(node); }

  // 새 대화
  btnStart.addEventListener("click", async () => {
    chatList.innerHTML = "";
    addMsg("bot", "안녕하세요! 편하게 일상을 얘기해 주세요.");
    addMsg("bot", "오늘 하루는 어떻게 보내셨어요?");
    recEl.value = "";
    resetTimer(60);
    try {
      const r = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset: true })
      });
      if (!r.ok) console.warn("reset not OK", r.status);
    } catch (e) {
      console.warn("reset error", e);
    }
  });

  // 녹음(추후 연결)
  btnRec.addEventListener("click", () => { recEl.focus(); });

  btnExt.addEventListener("click", () => {
    secondsLeft += 30;
    timeEl.textContent = `남은 시간 ${secondsLeft}s`;
  });

  // 보내기
  btnSend.addEventListener("click", async () => {
    const text = (recEl.value || "").trim();
    if (!text) return;
    addMsg("user", text);
    recEl.value = "";

    const typing = addTyping();
    try {
      const r = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!r.ok) {
        const bodyText = await r.text().catch(() => "");
        throw new Error(`HTTP ${r.status} ${r.statusText} – ${bodyText.slice(0, 200)}`);
      }

      const raw = await r.text();
      let data;
      try {
        data = JSON.parse(raw);
      } catch (e) {
        throw new Error("Invalid JSON: " + raw.slice(0, 200));
      }

      removeTyping(typing);
      addMsg("bot", (data && data.reply) ? data.reply : "응답을 준비하고 있어요.");
    } catch (e) {
      removeTyping(typing);
      addMsg("bot", "서버 연결에 문제가 있어요. 잠시 후 다시 시도해 주세요.");
      console.error(`[${API}] fetch error:`, e);
    }
    resetTimer(60);
  });
});
