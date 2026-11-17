// Fake game used by the recorder tests. Mirrors the real page structure (see
// tools/recorder/selectors.py): tokens are span#token-title-N / span#token-text-N where N is the
// character offset in the title / body, hidden ones have class "obfuscated" and block
// characters, a guess adds a row to the guess list (found / not-found / warning) and the tokens
// it matched get class "guess-match". Guesses are answered after a 50 ms delay. The send button
// has type="button" like the real one.
(async () => {
  const STOPWORDS = new Set(["de", "a", "o", "e"]);
  const TOKEN_RE = /[\p{L}0-9]+/gu;
  const HEADER_RE = /^=+\s.*\s=+$/;
  const BLOCK = "\u2588";

  const tokens = [];
  const seen = new Set();
  const history = [];
  const rows = []; // counted guesses, newest first
  let warning = null;

  const input = document.getElementById("guess-input");
  const form = document.querySelector("#app-controls-section form");
  const button = form.querySelector("button");
  const list = document.getElementById("guess-list");

  function renderLine(container, kind, line, base) {
    let last = 0;
    for (const match of line.matchAll(TOKEN_RE)) {
      if (match.index > last) {
        container.appendChild(document.createTextNode(line.slice(last, match.index)));
      }
      const span = document.createElement("span");
      const length = Array.from(match[0]).length;
      span.id = "token-" + kind + "-" + (base + match.index);
      span.setAttribute("data-token-length", String(length));
      span.className = "token";
      container.appendChild(span);
      tokens.push({ el: span, text: match[0], length: length });
      last = match.index + match[0].length;
    }
    if (last < line.length) {
      container.appendChild(document.createTextNode(line.slice(last)));
    }
  }

  function hide(token) {
    token.el.classList.add("obfuscated");
    token.el.textContent = BLOCK.repeat(token.length);
  }

  function reveal(token) {
    token.el.classList.remove("obfuscated");
    token.el.textContent = token.text;
  }

  function renderList() {
    const total = rows.length;
    let html = "<ul><li><span>" + total + (total === 1 ? " palpite" : " palpites") + "</span></li>";
    if (warning) {
      html +=
        '<li><div class="guess-warning list-item-left">' + warning.guess +
        '</div><div class="guess-warning list-item-right">' + warning.message + "</div></li>";
    }
    rows.forEach((row, index) => {
      const status = row.count > 0 ? "guess-found" : "guess-not-found";
      html +=
        '<li><div class="list-item' + (index === 0 ? " Mui-selected" : "") + '">' +
        '<div class="' + status + ' list-item-left">' + row.guess +
        '</div><div class="' + status + ' list-item-right">' + row.count + "</div></div></li>";
    });
    list.innerHTML = html + "</ul>";
  }

  function saveState() {
    localStorage.setItem("history", JSON.stringify(history));
  }

  await window.fetch("/asset-manifest.json").then((r) => r.json());
  const thirdParty = await window.fetch("/third-party-url").then((r) => r.text());
  window.fetch(thirdParty + "/pixel.json").catch(() => {});
  const article = await window.fetch("/articles/fake.json").then((r) => r.json());

  renderLine(document.getElementById("article-title"), "title", article.title, 0);
  const body = document.getElementById("article-text");
  let offset = 0;
  for (const line of article.text.split("\n")) {
    if (line.trim()) {
      if (HEADER_RE.test(line)) {
        const heading = document.createElement("h2");
        body.appendChild(heading);
        renderLine(heading, "text", line, offset);
      } else {
        const paragraph = document.createElement("p");
        body.appendChild(paragraph);
        renderLine(paragraph, "text", line, offset);
      }
    }
    offset += line.length + 1;
  }
  for (const token of tokens) {
    if (STOPWORDS.has(token.text.toLowerCase())) reveal(token);
    else hide(token);
  }
  renderList();

  // Assembled at runtime so the literal secret is not in this file (which the recorder saves).
  localStorage.setItem("authToken", ["PLANTED", "STORAGE", "SECRET", "456"].join("-"));
  saveState();
  console.log("game ready");

  function submit() {
    const guess = input.value.trim().toLowerCase();
    input.value = "";
    if (!guess) return;
    let outcome = "ok";
    if (STOPWORDS.has(guess)) outcome = "stopword";
    else if (seen.has(guess)) outcome = "repeat";
    else seen.add(guess);
    setTimeout(() => {
      warning = null;
      if (outcome === "stopword") {
        warning = { guess: guess, message: "não conta" };
      } else if (outcome === "repeat") {
        warning = { guess: guess, message: "já foi" };
      } else {
        let count = 0;
        for (const token of tokens) {
          const hit = token.text.toLowerCase() === guess;
          token.el.classList.toggle("guess-match", hit);
          if (hit) {
            reveal(token);
            count += 1;
          }
        }
        rows.unshift({ guess: guess, count: count });
      }
      renderList();
      history.push({ guess, outcome });
      saveState();
    }, 50);
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  button.addEventListener("click", submit);
})();
