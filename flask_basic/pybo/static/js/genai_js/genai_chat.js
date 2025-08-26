// pybo/static/js/genai_playground.js
document.addEventListener("DOMContentLoaded", () => {
  // ====== 게스트 자동 초기화(비콘) 가드 ======
  let isSubmitting = false;
  let didReset = false;

  const isGuest = (() => {
    if (typeof window.GENAI_IS_GUEST !== "undefined") return !!window.GENAI_IS_GUEST;
    const html = document.documentElement;
    if (html?.dataset && typeof html.dataset.isGuest !== "undefined") {
      const v = html.dataset.isGuest;
      return v === "1" || v === "true";
    }
    const meta = document.querySelector('meta[name="is-guest"]');
    if (meta) return meta.content === "1" || meta.content === "true";
    const hidden = document.getElementById("is_guest");
    if (hidden) return hidden.value === "1" || hidden.value === "true";
    return false;
  })();

  const RESET_URL = "/genai/reset_anon";
  function safeReset() {
    if (!isGuest || isSubmitting || didReset) return;
    try { navigator.sendBeacon(RESET_URL); didReset = true; } catch (_) {}
  }

  // ====== 요소 참조(여러 템플릿을 커버하도록 유연하게) ======
  const form = document.getElementById("chatForm") || document.querySelector("form");
  const promptEl =
    document.getElementById("prompt") ||
    document.getElementById("message") ||
    document.querySelector("textarea[name=message], textarea#message, textarea#prompt");
  const outEl =
    document.getElementById("output") ||
    document.getElementById("chatLog") ||
    document.querySelector("#log, #output");
  const runBtn =
    document.getElementById("runBtn") ||
    document.getElementById("sendBtn") ||
    document.querySelector("button#runBtn, button#sendBtn") ||
    null;

  // 봇 API로 고정 (채팅 페이지로 날아가지 않도록)
  const API_URL = "/genai/bot/api/message";

  function val(id) {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) : NaN;
  }

  async function generate() {
    const prompt = (promptEl?.value || "").trim();
    if (!prompt) { alert("프롬프트를 입력해 주세요."); return; }

    if (outEl) outEl.textContent = "생성 중…";
    if (runBtn) runBtn.disabled = true;
    isSubmitting = true; // 페이지 이탈 비콘 방지

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text: prompt })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (outEl) outEl.textContent = (data.reply || "(응답이 비어 있습니다)").toString();
    } catch (err) {
      if (outEl) outEl.textContent = "❌ " + (err.message || String(err));
    } finally {
      if (runBtn) runBtn.disabled = false;
      isSubmitting = false;
    }
  }

  // --- 핵심: 폼 submit 막기(액션으로 날아가지 않도록) ---
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      e.stopPropagation();
      generate();
      return false;
    }, true);
  }

  // 버튼 클릭도 기본 submit 막기
  if (runBtn) {
    // 버튼이 submit 타입이면 동작 전에 막음
    runBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      generate();
    });
  }

  // Ctrl/Cmd+Enter: 실행
  if (promptEl) {
    promptEl.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        generate();
      }
    });
  }

  // 페이지 떠날 때만 초기화 비콘
  window.addEventListener("pagehide", safeReset);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") safeReset();
  });
});
