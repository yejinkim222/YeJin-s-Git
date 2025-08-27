// static/js/disease_voice.js
document.addEventListener("DOMContentLoaded", () => {
  console.log("[disease] JS loaded");

  // --- DOM ---
  const box        = document.getElementById("screenBox");
  const API        = box?.dataset?.api || "/disease/ad8/step";

  const logEl      = document.getElementById("log");
  const answerBox  = document.getElementById("answerBox");
  const startBtn   = document.getElementById("startBtn");
  const recBtn     = document.getElementById("recBtn");
  const extendBtn  = document.getElementById("extendBtn");
  const nextBtn    = document.getElementById("nextBtn");
  const timeLeftEl = document.getElementById("timeLeft");

  if (!logEl || !answerBox || !startBtn || !recBtn || !extendBtn || !nextBtn || !timeLeftEl) {
    console.error("[disease] 필수 요소가 없습니다.");
    alert("스크리닝 UI 요소를 찾지 못했습니다. 템플릿 id를 확인하세요.");
    return;
  }

  // --- 공통 유틸 ---
  const escapeHTML = (s) => s.replace(/[&<>"']/g, m => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]
  ));

  // 말풍선 형태로 메시지 추가 (assistant | user)
  function appendMsg(role, text){
    const cls = role === 'user' ? 'user' : 'assistant';
    logEl.insertAdjacentHTML(
      'beforeend',
      `<div class="msg-row ${cls}">
         <div class="msg-bubble">${escapeHTML(String(text || ''))}</div>
       </div>`
    );
    logEl.scrollTop = logEl.scrollHeight;
  }

  // 초기화(로그 리셋 + 안내문)
  function resetLogWithIntro(){
    logEl.innerHTML = '';
    appendMsg('assistant', '간단한 문답으로 최근 인지 상태를 살펴봅니다. 의료 진단이 아니며, 결과는 참고용입니다.');
  }

  // --- 상태/타이머 ---
  let timeLeft = 60;
  let timerId = null;
  let recognizing = false;
  let sr = null; // SpeechRecognition

  function stopTimer(){ if (timerId) { clearInterval(timerId); timerId = null; } }
  function renderTime(){ timeLeftEl.textContent = String(timeLeft); }
  function startTimer(){
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
  function initSROnce(){
    if (sr) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      appendMsg('assistant', '이 브라우저에서는 음성 인식을 지원하지 않습니다. 직접 입력해 주세요.');
      recBtn.disabled = true;
      return;
    }
    sr = new SR();
    sr.lang = 'ko-KR';
    sr.interimResults = false;
    sr.maxAlternatives = 1;

    sr.onstart = () => { recognizing = true;  recBtn.textContent = '정지'; };
    sr.onend   = () => { recognizing = false; recBtn.textContent = '녹음'; stopTimer(); };
    sr.onerror = (e) => {
      recognizing = false;
      recBtn.textContent = '녹음';
      stopTimer();
      console.error('SpeechRecognition error:', e);
      appendMsg('assistant', '음성 인식 중 문제가 발생했어요. 텍스트로 입력하셔도 됩니다.');
    };
    sr.onresult = (e) => {
      if (!e.results?.[0]?.[0]) return;
      answerBox.value = e.results[0][0].transcript || '';
    };
  }
  function startSR(){
    initSROnce();
    if (!sr || recognizing) return;
    try { sr.start(); startTimer(); }
    catch (e) { console.error(e); appendMsg('assistant', '음성 인식을 시작할 수 없어요. 직접 입력을 이용해 주세요.'); }
  }
  function stopSR(){ if (sr && recognizing) { try { sr.stop(); } catch(_) {} } }

  // --- 이벤트 (중복 없음!) ---
  startBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ reset: true }),
        cache: 'no-store',
      });
      const data = await res.json().catch(() => ({}));
      resetLogWithIntro();
      if (data?.reply) appendMsg('assistant', data.reply);
      else appendMsg('assistant', '준비가 완료됐어요. 시작해 볼까요?');
    } catch (err) {
      console.warn('reset call failed:', err);
      resetLogWithIntro();
      appendMsg('assistant', '초기화에 실패했어요. 새로고침 후 다시 시도해 주세요.');
    }
    stopTimer();
    recognizing = false;
    if (sr) { try { sr.abort(); } catch(_) {} }
    timeLeft = 60; renderTime();
    answerBox.value = '';
    nextBtn.disabled = false;
    recBtn.disabled = false;
    recBtn.textContent = '녹음';
  });

  recBtn.addEventListener('click', (e) => {
    e.preventDefault();
    if (recognizing) stopSR(); else startSR();
  });

  extendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    timeLeft = Math.min(timeLeft + 30, 600);
    renderTime();
  });

  nextBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    const userText = (answerBox.value || '').trim();
    if (!userText) { appendMsg('assistant', '답변이 비어 있어요. 한두 문장으로 편하게 적어 주세요.'); return; }

    appendMsg('user', userText);
    answerBox.value = '';
    stopTimer(); stopSR();
    nextBtn.disabled = true;

    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ text: userText }),
        cache: 'no-store',
      });
      const mime = res.headers.get('Content-Type') || '';
      const data = mime.includes('application/json') ? await res.json() : { reply: await res.text() };

      if (!res.ok) {
        console.error('HTTP error', res.status, data);
        appendMsg('assistant', `서버 오류(${res.status})`);
      } else {
        const reply = (data && data.reply) ? String(data.reply) : '처리에 실패했어요. 잠시 후 다시 시도해 주세요.';
        appendMsg('assistant', reply);

        if (data && data.done) {
          nextBtn.disabled = true;
          recBtn.disabled  = true;
          appendMsg('assistant', '대화를 마무리했습니다. [시작]을 누르면 다시 진행할 수 있어요.');
        } else {
          nextBtn.disabled = false;
        }
      }
    } catch (err) {
      console.error('fetch failed:', err);
      appendMsg('assistant', '네트워크 오류로 답변을 받지 못했어요. 다시 시도해 주세요.');
      nextBtn.disabled = false;
    }
  });

  // 초기 시간 표시
  timeLeft = 60;
  renderTime();
});
