(function(){
  const $ = (s, el=document) => el.querySelector(s);
  const app = $("#genaiApp");
  const API = {
    message: app.dataset.apiMessage,
    reset:   app.dataset.apiReset,
    summary: app.dataset.apiSummary,
  };

  const chatList = $("#chatList");
  const modeBadge = $("#modeBadge");
  const prompt = $("#prompt");
  const btnSend = $("#btnSend");
  const btnReset = $("#btnReset");
  const btnMic = $("#btnMic");
  const timerEl = $("#timer");

  // ---------- UI ----------
  function esc(s){
    return (s||"").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }
  function addMsg(role, text){
    const li = document.createElement("li");
    li.className = "msg " + (role === "user" ? "user" : "bot");
    li.innerHTML = `<div class="bubble ${role === "user" ? "user" : "bot"}">${esc(text)}</div>`;
    chatList.appendChild(li);
    chatList.parentElement.scrollTop = chatList.parentElement.scrollHeight;
  }

  // ---------- Send ----------
  let busy = false;
  async function sendText(text){
    if (busy) return;
    text = (text||"").trim();
    if (!text) return;

    busy = true;
    addMsg("user", text);
    prompt.value = "";
    try{
      const r = await fetch(API.message, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text}),
        cache: "no-store",
      });
      const data = await r.json();
      const reply = (data && data.reply) ? String(data.reply) : "답변을 받지 못했어요.";
      addMsg("bot", reply);

      if (data && data.meta){
        setModeBadge(data.meta.mode);
        if (data.meta.done){
          btnSend.disabled = true;
          btnMic.disabled = true;
          stopSR();
          stopTimer();
          timerEl.textContent = "";
        }
      }
    }catch(e){
      addMsg("bot","네트워크 문제가 있어요. 잠시 후 다시 시도해 주세요.");
    }finally{
      busy = false;
    }
  }

  btnSend.addEventListener("click", () => sendText(prompt.value));
  prompt.addEventListener("keydown", (e)=>{
    if (e.key === "Enter"){
      e.preventDefault();
      sendText(prompt.value);
    }
  });

  // ---------- Reset ----------
  btnReset.addEventListener("click", async ()=>{
    try{
      await fetch(API.reset, {method:"POST"});
      location.reload();
    }catch(_){}
  });

  // ---------- Voice (Web Speech) ----------
  let sr = null, recognizing = false;
  let timeLeft = 0, tId = null;

  function stopTimer(){
    if (tId){ clearInterval(tId); tId=null; }
  }
  function startTimer(sec=60){
    stopTimer();
    timeLeft = sec;
    timerEl.textContent = `녹음 ${timeLeft}s`;
    tId = setInterval(()=>{
      timeLeft = Math.max(0, timeLeft-1);
      timerEl.textContent = `녹음 ${timeLeft}s`;
      if (timeLeft === 0){
        stopTimer();
        stopSR(true); // timeout=true -> 자동 전송
      }
    }, 1000);
  }

  function initSR(){
    if (sr) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR){
      btnMic.disabled = true;
      addMsg("bot","이 브라우저에서는 음성 인식을 지원하지 않습니다. 텍스트 입력을 이용해 주세요.");
      return;
    }
    sr = new SR();
    sr.lang = "ko-KR";
    sr.interimResults = false;
    sr.maxAlternatives = 1;

    sr.onstart = ()=>{
      recognizing = true;
      btnMic.textContent = "정지";
      startTimer(60); // 연장 없이 60초 고정
    };
    sr.onend = ()=>{
      recognizing = false;
      btnMic.textContent = "녹음";
      stopTimer();
    };
    sr.onerror = (e)=>{
      recognizing = false;
      btnMic.textContent = "녹음";
      stopTimer();
      addMsg("bot","음성 인식 중 문제가 발생했어요. 텍스트로 입력해 주세요.");
      console.warn(e);
    };
    sr.onresult = (e)=>{
      try{
        const txt = e.results?.[0]?.[0]?.transcript || "";
        if (txt.trim()){
          prompt.value = txt.trim();
        }
      }catch(_){}
    };
  }

  function startSR(){
    initSR();
    if (!sr || recognizing) return;
    try{ sr.start(); }catch(_){}
  }
  function stopSR(timeout=false){
    if (sr && recognizing){
      try{ sr.stop(); }catch(_){}
    }
    if (timeout){
      const v = (prompt.value||"").trim();
      sendText(v || "(무응답)");
    }
  }

  btnMic.addEventListener("click", ()=>{
    if (recognizing) stopSR(false);
    else startSR();
  });

  // ---------- unload 시 요약 저장(로그인 사용자만 서버에서 저장) ----------
  window.addEventListener("beforeunload", ()=>{
    try{
      const payload = new Blob([JSON.stringify({ts: Date.now()})], {type: "application/json"});
      navigator.sendBeacon(API.summary, payload);
    }catch(_){}
  });

})();
