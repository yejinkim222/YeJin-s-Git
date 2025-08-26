document.addEventListener("DOMContentLoaded", () => {
  console.log("[disease] JS loaded");

  // --- DOM ---
  const box        = document.getElementById("screenBox");
  // ★ 기본도 AD8로
  const API        = box?.dataset?.api || "/disease/ad8/step";

  const logEl      = document.getElementById("log");
  const answerBox  = document.getElementById("answerBox");
  const startBtn   = document.getElementById("startBtn");
  const recBtn     = document.getElementById("recBtn");
  const extendBtn  = document.getElementById("extendBtn");
  const nextBtn    = document.getElementById("nextBtn");
  const timeLeftEl = document.getElementById("timeLeft");

  if (!logEl || !answerBox || !startBtn || !recBtn || !extendBtn || !nextBtn || !timeLeftEl) {
    console.error("[disease] 필수 요소가 없습니다.", {logEl, answerBox, startBtn, recBtn, extendBtn, nextBtn, timeLeftEl});
    alert("스크리닝 UI 요소를 찾지 못했습니다. 템플릿 id를 확인하세요.");
    return;
  }

  // --- 상태 ---
  let timeLeft = 60;
  let timerId = null;
  let recognizing = false;
  let sr = null; // SpeechRecognition

  // --- 로그 유틸 ---
  function addRow(role, text) {
    const row = document.createElement("div");
    row.className = "msg-row";
    const roleSpan = document.createElement("span");
    roleSpan.className = "role";
    roleSpan.textContent = role === "user" ? "나:" : "SolCare:";
    row.appendChild(roleSpan);
    row.appendChild(document.createTextNode(" " + (text || "")));
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function resetLogWithIntro() {
    logEl.innerHTML = "";
    addRow("assistant", "간단한 문답으로 최근 인지 상태를 살펴봅니다. 의료 진단이 아니며, 결과는 참고용입니다.");
  }

  // --- 타이머 ---
  function stopTimer() { if (timerId) { clearInterval(timerId); timerId = null; } }
  function renderTime() { timeLeftEl.textContent = String(timeLeft); }
  function startTimer() {
    stopTimer();
    timeLeft = Math.max(1, timeLeft);
    renderTime();
    timerId = setInterval(() => {
      timeLeft = Math.max(0, timeLeft - 1);
      renderTime();
      if (timeLeft === 0) { stopTimer(); stopSR(); }
    }, 1000);
  }

  // --- 음성 인식(Web Speech API) ---
  function initSROnce() {
    if (sr) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      addRow("assistant", "이 브라우저에서는 음성 인식을 지원하지 않습니다. 직접 입력해 주세요.");
      recBtn.disabled = true;
      return;
    }
    sr = new SR();
    sr.lang = "ko-KR";
    sr.interimResults = false;
    sr.maxAlternatives = 1;

    sr.onstart = () => { recognizing = true; recBtn.textContent = "정지"; };
    sr.onend   = () => { recognizing = false; recBtn.textContent = "녹음"; stopTimer(); };
    sr.onerror = (e) => {
      recognizing = false;
      recBtn.textContent = "녹음";
      stopTimer();
      console.error("SpeechRecognition error:", e);
      addRow("assistant", "음성 인식 중 문제가 발생했어요. 텍스트로 입력해 주셔도 됩니다.");
    };
    sr.onresult = (e) => {
      if (!e.results || !e.results[0] || !e.results[0][0]) return;
      const txt = e.results[0][0].transcript || "";
      answerBox.value = txt;
    };
  }
  function startSR() {
    initSROnce();
    if (!sr || recognizing) return;
    try { sr.start(); startTimer(); }
    catch (e) { console.error(e); addRow("assistant", "음성 인식을 시작할 수 없어요. 직접 입력을 이용해 주세요."); }
  }
  function stopSR() { if (sr && recognizing) { try { sr.stop(); } catch(_) {} } }

  // --- 이벤트 바인딩 ---
  // 시작: 서버에 reset 보내고, 서버가 내려준 첫 질문(reply)을 바로 표시
  startBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ reset: true }),
        cache: "no-store",
      });
      const data = await res.json().catch(() => ({}));
      resetLogWithIntro();
      if (data && data.reply) addRow("assistant", data.reply);
      else addRow("assistant", "준비가 완료됐어요. 시작해 볼까요?");
    } catch (err) {
      console.warn("reset call failed:", err);
      resetLogWithIntro();
      addRow("assistant", "초기화에 실패했어요. 새로고침 후 다시 시도해 주세요.");
    }
    stopTimer();
    recognizing = false;
    if (sr) { try { sr.abort(); } catch(_) {} }
    timeLeft = 60; renderTime();
    answerBox.value = "";
    nextBtn.disabled = false;
    recBtn.disabled = false;
    recBtn.textContent = "녹음";
  });

  recBtn.addEventListener("click", (e) => {
    e.preventDefault();
    if (recognizing) stopSR(); else startSR();
  });

  extendBtn.addEventListener("click", (e) => {
    e.preventDefault();
    timeLeft = Math.min(timeLeft + 30, 600);
    renderTime();
  });

  // 다음: 답변 전송 → 서버가 주는 다음 질문(reply) 또는 최종 요약(reply) 표시
  nextBtn.addEventListener("click", async (e) => {
    e.preventDefault();

    const userText = (answerBox.value || "").trim();
    if (!userText) { addRow("assistant", "답변이 비어 있어요. 한두 문장으로 편하게 적어 주세요."); return; }

    addRow("user", userText);
    answerBox.value = "";
    stopTimer(); stopSR();
    nextBtn.disabled = true;

    try {
      const res = await fetch(API, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text: userText }),
        cache: "no-store",
      });

      const mime = res.headers.get("Content-Type") || "";
      const data = mime.includes("application/json") ? await res.json() : { reply: await res.text() };

      if (!res.ok) {
        console.error("HTTP error", res.status, data);
        addRow("assistant", `서버 오류(${res.status})`);
      } else {
        const reply = (data && data.reply) ? String(data.reply) : "처리에 실패했어요. 잠시 후 다시 시도해 주세요.";
        addRow("assistant", reply);

        // ★ AD8 플로우는 done 플래그로 종료 여부를 알려줌
        if (data && data.done) {
          nextBtn.disabled = true;
          recBtn.disabled  = true;
          addRow("assistant", "대화를 마무리했습니다. [시작]을 누르면 다시 진행할 수 있어요.");
        } else {
          nextBtn.disabled = false;
        }
      }
    } catch (err) {
      console.error("fetch failed:", err);
      addRow("assistant", "네트워크 오류로 답변을 받지 못했어요. 다시 시도해 주세요.");
      nextBtn.disabled = false;
    }
  });

  // 초기 표시
  // 안내만 띄우고, 실제 첫 질문은 [시작] 누르면 서버에서 받아서 출력
  timeLeft = 60;
  renderTime();
});
