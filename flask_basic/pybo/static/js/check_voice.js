// 인지 스크리닝(음성) – VAD + 타이머 + 연장 + 편집 가능 답안 + 최종 점수 계산/출력
// 서버 필요: POST /genai/asr  (form-data: file=audio.webm) → {text:"..."}

(() => {
  // ====== 설정값 ======
  const DEFAULT_MAX_SEC   = 60;   // 질문당 기본 최대 녹음 시간
  const EXTEND_SEC        = 30;   // 연장 1회당 추가 시간
  const SILENCE_THRESHOLD = 0.01; // VAD 임계치(0~1)
  const SILENCE_HANG_MS   = 1200; // 이 시간 이상 무음 → 자동 종료
  const MIN_SPEECH_MS     = 400;  // 최소 발화 보장

  // ====== 요소 ======
  const logEl      = document.getElementById("log");
  const answerBox  = document.getElementById("answerBox");
  const startBtn   = document.getElementById("startBtn");
  const recBtn     = document.getElementById("recBtn");
  const extendBtn  = document.getElementById("extendBtn");
  const nextBtn    = document.getElementById("nextBtn");
  const timeLeftEl = document.getElementById("timeLeft");

  // ====== 상태 ======
  let media, recorder, chunks = [];
  let audioCtx, srcNode, analyser, dataArray;
  let vadTimer = null, hardStopTimer = null, uiTimer = null;
  let startedAt = 0, lastNonSilence = 0;
  let maxSec = DEFAULT_MAX_SEC;
  let questionIdx = 0;        // 0: 날짜 응답, 1..N: script 질문
  const answers = [];         // 사용자의 각 답변 문자열 저장

  // 질문 시나리오(날짜 질문은 log에 미리 고정되어 있으므로 배열엔 포함 X)
  const SCRIPT = [
    "오늘은 무슨 요일인가요? (예: 금요일)",
    "지금은 무슨 계절인가요? (봄/여름/가을/겨울)",
    "100에서 7씩 두 번 빼 주세요. (예: 93 86 79)",
    "아래 문장을 그대로 말해 주세요: “봄에는 꽃이 핀다”",
  ];

  // ====== 도우미 ======
  function addBot(text) {
    const row = document.createElement("div");
    row.className = "msg-row";
    row.innerHTML = `<span class="role">SolCare:</span> ${escapeHtml(text)}`;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function addUser(text) {
    const row = document.createElement("div");
    row.className = "msg-row";
    row.innerHTML = `<span class="role">나:</span> ${escapeHtml(text)}`;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function addResult(html) {
    const row = document.createElement("div");
    row.className = "msg-row";
    row.innerHTML = `<span class="role">SolCare:</span> ${html}`;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function escapeHtml(s) {
    return (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function setCountdown(sec) { timeLeftEl.textContent = Math.max(0, Math.ceil(sec)); }
  function resetTimers() { try{clearInterval(vadTimer)}catch{}; try{clearInterval(uiTimer)}catch{}; try{clearTimeout(hardStopTimer)}catch{}; vadTimer=uiTimer=hardStopTimer=null; }
  function closeAudio() { if (audioCtx) { try{audioCtx.close()}catch{} } audioCtx = srcNode = analyser = dataArray = null; }
  function uiStateRecording(on) { extendBtn.disabled = !on; recBtn.textContent = on ? "정지" : "녹음"; }
  function resetPerQuestionUI() { maxSec = DEFAULT_MAX_SEC; setCountdown(maxSec); extendBtn.disabled = true; answerBox.value = ""; }

  // ====== 녹음/분석 ======
  async function startRec() {
    if (!media) media = await navigator.mediaDevices.getUserMedia({audio:true});
    chunks = [];

    recorder = new MediaRecorder(media, { mimeType:"audio/webm" });
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = onStop;

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    srcNode  = audioCtx.createMediaStreamSource(media);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    srcNode.connect(analyser);
    dataArray = new Uint8Array(analyser.fftSize);

    startedAt = Date.now();
    lastNonSilence = startedAt;

    vadTimer = setInterval(checkSilence, 60);
    uiTimer  = setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      setCountdown(maxSec - elapsed);
    }, 200);
    hardStopTimer = setTimeout(stopRec, maxSec * 1000);

    recorder.start();
    uiStateRecording(true);
  }

  function checkSilence() {
    analyser.getByteTimeDomainData(dataArray);
    let sum = 0;
    for (let i=0;i<dataArray.length;i++) {
      const v = (dataArray[i]-128)/128;
      sum += v*v;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    const now = Date.now();

    if (rms > SILENCE_THRESHOLD) lastNonSilence = now;

    const voicedMs = lastNonSilence - startedAt;
    const silentMs = now - lastNonSilence;

    if (voicedMs > MIN_SPEECH_MS && silentMs > SILENCE_HANG_MS) stopRec();
  }

  function stopRec() {
    resetTimers();
    if (recorder && recorder.state !== "inactive") recorder.stop();
    uiStateRecording(false);
    closeAudio();
  }

  async function onStop() {
    if (!chunks.length) return;
    const blob = new Blob(chunks, { type: "audio/webm" });
    const fd = new FormData();
    fd.append("file", blob, "answer.webm");

    try {
      const res = await fetch("/genai/asr", { method:"POST", body: fd });
      if (!res.ok) throw new Error("ASR 서버 오류");
      const data = await res.json();
      const text = (data && data.text) ? data.text : "";
      answerBox.value = text;
    } catch (err) {
      console.error(err);
      answerBox.value = "(인식 실패) 내용이 비었거나 서버 오류입니다. 직접 입력해 주세요.";
    }
  }

  // ====== 점수 계산 ======
  function normalize(s) {
    return (s || "")
      .replace(/[^\p{L}\p{N} ]/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function scoreDate(ans) {
    // 정답: 오늘 날짜 (로컬)
    const d   = new Date();
    const yyyy = d.getFullYear();
    const mm   = String(d.getMonth()+1).padStart(2,"0");
    const dd   = String(d.getDate()).padStart(2,"0");
    const ymd  = `${yyyy}-${mm}-${dd}`;

    const t = ans || "";
    const hits = [
      t.includes(String(yyyy)),
      t.includes(`${Number(mm)}`) || t.includes(`${mm}`) || t.includes("월"),
      t.includes(`${Number(dd)}`) || t.includes(`${dd}`) || t.includes("일"),
      t.includes(ymd)
    ].filter(Boolean).length;

    // 월/일 둘 다 맞거나 YYYY-MM-DD가 포함되면 2점, 하나만 맞으면 1점
    if (t.includes(ymd) || (t.match(/월/) && t.match(/일/))) return 2;
    if (hits >= 2) return 2;
    if (hits >= 1) return 1;
    return 0;
  }

  function scoreWeekday(ans) {
    const wd = ["일","월","화","수","목","금","토"][new Date().getDay()];
    return (ans || "").includes(`${wd}요일`) || (ans || "").includes(wd) ? 2 : 0;
  }

  function scoreSeason(ans) {
    const m = new Date().getMonth() + 1;
    const season = (m>=3 && m<=5) ? "봄" : (m<=8) ? "여름" : (m<=11) ? "가을" : "겨울";
    return (ans || "").includes(season) ? 2 : 0;
  }

  function scoreSerialSevens(ans) {
    // 목표 시퀀스 일부라도 올바른 순서로 포함되면 부분점수
    const seq = [93, 86, 79, 72, 65];
    const nums = (ans.match(/-?\d+/g) || []).map(n => parseInt(n,10));
    let i = 0, hit = 0;
    for (const n of nums) {
      if (n === seq[i]) { hit++; i++; if (i >= seq.length) break; }
    }
    if (hit >= 3) return 2;  // 93,86,79까지 맞추면 만점
    if (hit >= 1) return 1;
    return 0;
  }

  function scoreRepeatPhrase(ans) {
    const target = "봄에는꽃이핀다";
    const nrm = (ans || "").replace(/\s+/g,"").replace(/[^\p{L}\p{N}]/gu,"");
    if (!nrm) return 0;
    if (nrm.includes(target)) return 2;
    // 느슨한 기준: 핵심 토큰 2개 이상 포함
    const hits = ["봄", "꽃", "핀"].filter(t => nrm.includes(t)).length;
    return hits >= 2 ? 1 : 0;
  }

  function computeTotalScore(ansList) {
    // ansList: [날짜, 요일, 계절, 7씩빼기, 문장따라말하기]
    const [a0, a1, a2, a3, a4] = ansList.map(a => (a||"").trim());
    const s0 = scoreDate(a0);
    const s1 = scoreWeekday(a1);
    const s2 = scoreSeason(a2);
    const s3 = scoreSerialSevens(a3);
    const s4 = scoreRepeatPhrase(a4);
    const total = s0+s1+s2+s3+s4;
    let band = "정상 범위로 보입니다.";
    if (total <= 4) band = "의심 소견이 있어 전문평가를 권합니다.";
    else if (total <= 7) band = "경계 범위입니다. 경과관찰을 권합니다.";
    return { total, breakdown: {s0,s1,s2,s3,s4}, band };
  }

  function showFinalResult() {
    const { total, breakdown, band } = computeTotalScore(answers);
    const html =
      `총점 <b>${total}/10</b> – ${band}<br>` +
      `<small>세부: 날짜 ${breakdown.s0}, 요일 ${breakdown.s1}, 계절 ${breakdown.s2}, 7씩빼기 ${breakdown.s3}, 문장따라말하기 ${breakdown.s4}</small>`;
    addResult(html);
    addResult(`<small>※ 교육·연구용 참고도구이며, 의학적 진단이 아닙니다.</small>`);
    // 필요 시 서버 저장
    // fetch('/genai/screen/finish', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ answers, total, breakdown })})
  }

  // ====== UI 이벤트 ======
  startBtn.addEventListener("click", () => {
    addBot("준비되면 ‘녹음’을 눌러 답해 주세요.");
    resetPerQuestionUI();
  });

  recBtn.addEventListener("click", () => {
    if (recorder && recorder.state === "recording") {
      stopRec();
    } else {
      resetPerQuestionUI();
      startRec();
    }
  });

  extendBtn.addEventListener("click", () => {
    if (!startedAt) return;
    const elapsed = (Date.now() - startedAt) / 1000;
    maxSec += EXTEND_SEC;
    setCountdown(maxSec - elapsed);
    try { clearTimeout(hardStopTimer); } catch(_) {}
    hardStopTimer = setTimeout(stopRec, (maxSec - elapsed) * 1000);
  });

  nextBtn.addEventListener("click", () => {
    // 녹음 중이면 먼저 정지
    if (recorder && recorder.state === "recording") { stopRec(); return; }

    const text = (answerBox.value || "").trim();
    if (!text) { alert("답변이 비어 있습니다. 녹음하거나 텍스트를 입력해 주세요."); return; }

    // 현재 질문에 대한 답변 저장
    answers[questionIdx] = text;
    addUser(text);

    // 다음 질문 세팅
    questionIdx++;

    // 시나리오: 0(날짜) 처리 후 1..SCRIPT.length 진행
    const nextIdx = questionIdx - 1; // SCRIPT 인덱스
    if (nextIdx < SCRIPT.length) {
      resetPerQuestionUI();
      addBot(SCRIPT[nextIdx]);
    } else {
      // 모든 답변 수집 완료 → 결과 출력
      showFinalResult();
      // 버튼 상태(원하면 비활성화)
      // nextBtn.disabled = true; recBtn.disabled = true; extendBtn.disabled = true;
    }
  });

})();
