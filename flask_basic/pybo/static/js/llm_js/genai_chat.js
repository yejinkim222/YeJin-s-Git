// pybo/static/js/genai_playground.js
document.addEventListener("DOMContentLoaded", () => {
  // ====== 게스트 자동 초기화(비콘) 가드 ======
  // 이 값이 true면 '페이지 떠날 때' 초기화 비콘을 보내지 않습니다.
  let isSubmitting = false;
  // 중복 비콘 방지
  let didReset = false;

  // 게스트 여부 탐지 (템플릿 수정 없이 최대한 유연하게)
  // 1) window.GENAI_IS_GUEST = true/false
  // 2) <html data-is-guest="1">
  // 3) <meta name="is-guest" content="1">
  // 4) <input type="hidden" id="is_guest" value="1">
  const isGuest = (() => {
    if (typeof window.GENAI_IS_GUEST !== "undefined") return !!window.GENAI_IS_GUEST;
    const html = document.documentElement;
    if (html && html.dataset && typeof html.dataset.isGuest !== "undefined") {
      const v = html.dataset.isGuest;
      return v === "1" || v === "true";
    }
    const meta = document.querySelector('meta[name="is-guest"]');
    if (meta) return meta.content === "1" || meta.content === "true";
    const hidden = document.getElementById("is_guest");
    if (hidden) return hidden.value === "1" || hidden.value === "true";
    return false;
  })();

  // 실제 초기화 호출(게스트만, 제출 중이 아니고, 한 번만 전송)
  const RESET_URL = "/genai/reset_anon";
  function safeReset() {
    if (!isGuest || isSubmitting || didReset) return;
    try {
      navigator.sendBeacon(RESET_URL);
      didReset = true;
    } catch (_) { /* ignore */ }
  }

  // 모든 폼 전송에 대해 가드 (일반 submit도 포함됨)
  document.addEventListener("submit", (e) => {
    if (e.target && e.target.tagName === "FORM") {
      isSubmitting = true;
      // 혹시 오류로 완료 이벤트가 안 오는 경우를 대비한 타임아웃(선택)
      setTimeout(() => { isSubmitting = false; }, 10000);
    }
  }, true);

  // 페이지를 떠날 때(또는 탭이 숨겨질 때)만 초기화 비콘 전송
  window.addEventListener("pagehide", safeReset);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") safeReset();
  });

  // ====== 생성 버튼(fetch) 로직 ======
  const promptEl = document.getElementById("prompt");
  const outEl = document.getElementById("output");
  const runBtn = document.getElementById("runBtn");

  function val(id) {
    return parseFloat(document.getElementById(id).value);
  }

  async function generate() {
    const prompt = (promptEl?.value || "").trim();
    if (!prompt) { alert("프롬프트를 입력해 주세요."); return; }

    outEl.textContent = "생성 중…";
    runBtn.disabled = true;

    // 🔒 fetch 동안에는 '제출 중' 상태로 간주하여 초기화 비콘이 나가지 않게 함
    isSubmitting = true;

    try {
      const res = await fetch("/genai/complete", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          prompt,
          gen_kwargs: {
            temperature: val("temperature"),
            top_p: val("top_p"),
            max_new_tokens: parseInt(document.getElementById("max_new_tokens").value, 10)
          }
        })
      });

      // 네트워크가 끝났으므로 가드 해제
      isSubmitting = false;

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      outEl.textContent = data.text || "(응답이 비어 있습니다)";
    } catch (err) {
      outEl.textContent = "❌ " + (err.message || String(err));
    } finally {
      runBtn.disabled = false;
      // 혹시 위 try 블록 중간에서 에러가 나면 isSubmitting이 해제되지 않을 수 있어 한 번 더 보정
      isSubmitting = false;
    }
  }

  if (runBtn) {
    runBtn.addEventListener("click", generate);
  }

  // Ctrl/Cmd+Enter: 실행, Shift+Enter: 줄바꿈
  if (promptEl) {
    promptEl.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        generate();
      }
    });
  }
});
