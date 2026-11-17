// Passive page-side capture (REQ-REC-001). Prepended by session.py with:
//   window.__artigoRecorderConfig = { gameContainer, guessInput, guessForm, sendButton, quiescenceMs };
// Observes input events and DOM mutations inside the game container and reports them through
// the exposed binding window.__artigoRecorder(type, payload). It never changes the page.
(() => {
  if (window.top !== window || window.__artigoRecorderInstalled) return;
  window.__artigoRecorderInstalled = true;

  const cfg = window.__artigoRecorderConfig || {};
  const QUIESCENCE_MS = cfg.quiescenceMs || 300;
  const TEXT_LIMIT = 80;

  let lastGuessValue = "";

  function report(type, payload) {
    try {
      const sent = window.__artigoRecorder(type, payload);
      if (sent && typeof sent.catch === "function") sent.catch(() => {});
    } catch (error) {
      // The binding is unavailable (page closing); nothing to do.
    }
  }

  function truncate(text) {
    return String(text == null ? "" : text).trim().slice(0, TEXT_LIMIT);
  }

  // Stable selector: the id when present, else a tag:nth-of-type path up to the nearest id.
  function describe(node) {
    if (!node) return "";
    if (node.nodeType !== 1) return describe(node.parentNode);
    const parts = [];
    let el = node;
    while (el && el.nodeType === 1) {
      if (el.id) {
        parts.unshift("#" + el.id);
        break;
      }
      const tag = el.tagName.toLowerCase();
      if (tag === "html" || tag === "body") {
        parts.unshift(tag);
        break;
      }
      let index = 1;
      for (let sib = el.previousElementSibling; sib; sib = sib.previousElementSibling) {
        if (sib.tagName === el.tagName) index += 1;
      }
      parts.unshift(tag + ":nth-of-type(" + index + ")");
      el = el.parentElement;
    }
    return parts.join(" > ");
  }

  function matches(el, selector) {
    try {
      return !!(el && el.matches && selector && el.matches(selector));
    } catch (error) {
      return false;
    }
  }

  function guessField() {
    return cfg.guessInput ? document.querySelector(cfg.guessInput) : null;
  }

  function inputValue(el) {
    return el && typeof el.value === "string" ? el.value : "";
  }

  let lastSubmit = { value: null, at: 0 };

  function reportSubmit(value, target) {
    lastSubmit = { value: value, at: performance.now() };
    report("input.submit", { value: value, target: target });
    // A guess opens a fresh quiet window: the next dom.quiescent must come after the page
    // reacted to THIS guess (or after QUIESCENCE_MS without any reaction, for a miss), never
    // from a timer armed by earlier, unrelated mutations.
    restartQuietWindow();
  }

  document.addEventListener(
    "click",
    (event) => {
      const el = event.target && event.target.nodeType === 1 ? event.target : null;
      report("input.click", { target: describe(el), text: truncate(el && el.textContent) });
      // The real page's send button has type="button", so clicking it fires no submit event:
      // the click itself is the guess (capture phase, before the page clears the field).
      const button = el && el.closest && cfg.sendButton ? el.closest(cfg.sendButton) : null;
      if (button) {
        const field = guessField();
        const value = inputValue(field) || lastGuessValue;
        if (value.trim()) reportSubmit(value, describe(button));
      }
    },
    true
  );

  document.addEventListener(
    "keydown",
    (event) => {
      const el = event.target && event.target.nodeType === 1 ? event.target : null;
      const payload = { code: event.code, target: describe(el) };
      if (matches(el, cfg.guessInput)) payload.value = inputValue(el);
      report("input.key", payload);
    },
    true
  );

  document.addEventListener(
    "input",
    (event) => {
      if (matches(event.target, cfg.guessInput)) lastGuessValue = inputValue(event.target);
    },
    true
  );

  document.addEventListener(
    "submit",
    (event) => {
      const form = event.target && event.target.nodeType === 1 ? event.target : null;
      let field = null;
      if (form && cfg.guessInput && form.querySelector) field = form.querySelector(cfg.guessInput);
      if (!field) field = guessField();
      const value = field ? inputValue(field) : lastGuessValue;
      // The page may also submit programmatically right after a click on the send button,
      // which was already reported: skip that duplicate.
      if (lastSubmit.value === value && performance.now() - lastSubmit.at < 100) return;
      reportSubmit(value, describe(form));
    },
    true
  );

  // --- DOM mutations, restricted to the game container --------------------------------------

  let pending = 0;
  let lastMutationAt = 0;
  let timer = null;

  function onQuiet() {
    timer = null;
    const count = pending;
    pending = 0;
    report("dom.quiescent", {
      after_ms: Math.round(performance.now() - lastMutationAt),
      mutations: count,
    });
  }

  function restartQuietWindow() {
    lastMutationAt = performance.now();
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(onQuiet, QUIESCENCE_MS);
  }

  function brief(node) {
    if (node.nodeType === 1) return describe(node);
    return "#text:" + truncate(node.data);
  }

  function onMutations(records) {
    for (const record of records) {
      const payload = { kind: record.type, target: describe(record.target) };
      if (record.type === "attributes") {
        payload.attribute = record.attributeName;
        payload.old = record.oldValue;
        payload.new = record.target.getAttribute(record.attributeName);
      } else if (record.type === "characterData") {
        payload.old = record.oldValue;
        payload.new = record.target.data;
      } else {
        payload.added = Array.prototype.map.call(record.addedNodes, brief);
        payload.removed = Array.prototype.map.call(record.removedNodes, brief);
      }
      report("dom.mutation", payload);
      pending += 1;
    }
    lastMutationAt = performance.now();
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(onQuiet, QUIESCENCE_MS);
  }

  function observe(container) {
    new MutationObserver(onMutations).observe(container, {
      attributes: true,
      attributeOldValue: true,
      characterData: true,
      characterDataOldValue: true,
      childList: true,
      subtree: true,
    });
  }

  function attach() {
    const container = cfg.gameContainer ? document.querySelector(cfg.gameContainer) : null;
    if (container) {
      observe(container);
      return true;
    }
    return false;
  }

  if (!attach()) {
    // The container does not exist yet: wait for it with a silent observer on the document.
    const waiter = new MutationObserver(() => {
      if (attach()) waiter.disconnect();
    });
    waiter.observe(document, { childList: true, subtree: true });
  }
})();
