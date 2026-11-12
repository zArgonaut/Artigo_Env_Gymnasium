// Fake game used by the recorder tests. Follows the DOM contract of the F1 plan:
// token = maximal run of Unicode letters and ASCII digits; title tokens first; "== X =="
// lines become <h2>; guesses reveal matching tokens after a 50 ms delay.
(async () => {
  const STOPWORDS = new Set(["de", "a", "o", "e"]);
  const TOKEN_RE = /[\p{L}0-9]+/gu;
  const HEADER_RE = /^=+\s.*\s=+$/;

  const tokens = [];
  const seen = new Set();
  const history = [];
  let count = 0;

  const input = document.getElementById("guess-input");
  const form = document.getElementById("guess-form");
  const counter = document.getElementById("guess-count");
  const message = document.getElementById("message");

  function renderLine(container, line) {
    let last = 0;
    for (const match of line.matchAll(TOKEN_RE)) {
      if (match.index > last) {
        container.appendChild(document.createTextNode(line.slice(last, match.index)));
      }
      const span = document.createElement("span");
      span.className = "token";
      span.setAttribute("data-token-length", String(Array.from(match[0]).length));
      container.appendChild(span);
      tokens.push({ el: span, text: match[0], revealed: false });
      last = match.index + match[0].length;
    }
    if (last < line.length) {
      container.appendChild(document.createTextNode(line.slice(last)));
    }
  }

  function reveal(token) {
    token.revealed = true;
    token.el.classList.remove("hidden");
    token.el.textContent = token.text;
  }

  function saveState() {
    localStorage.setItem("history", JSON.stringify(history));
  }

  await window.fetch("/asset-manifest.json").then((r) => r.json());
  const thirdParty = await window.fetch("/third-party-url").then((r) => r.text());
  window.fetch(thirdParty + "/pixel.json").catch(() => {});
  const article = await window.fetch("/articles/fake.json").then((r) => r.json());

  renderLine(document.getElementById("title"), article.title);
  const body = document.getElementById("body");
  for (const line of article.text.split("\n")) {
    if (!line.trim()) continue;
    if (HEADER_RE.test(line)) {
      const heading = document.createElement("h2");
      body.appendChild(heading);
      renderLine(heading, line.replace(/^=+\s+|\s+=+$/g, ""));
    } else {
      const paragraph = document.createElement("p");
      body.appendChild(paragraph);
      renderLine(paragraph, line);
    }
  }
  for (const token of tokens) {
    if (STOPWORDS.has(token.text.toLowerCase())) reveal(token);
    else token.el.classList.add("hidden");
  }

  // Assembled at runtime so the literal secret is not in this file (which the recorder saves).
  localStorage.setItem("authToken", ["PLANTED", "STORAGE", "SECRET", "456"].join("-"));
  saveState();
  console.log("game ready");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const guess = input.value.trim().toLowerCase();
    input.value = "";
    if (!guess) return;
    let outcome = "ok";
    if (STOPWORDS.has(guess)) outcome = "stopword";
    else if (seen.has(guess)) outcome = "repeat";
    else seen.add(guess);
    setTimeout(() => {
      message.textContent = "";
      if (outcome === "stopword") {
        message.textContent = "não conta";
      } else if (outcome === "repeat") {
        message.textContent = "já foi";
      } else {
        count += 1;
        counter.textContent = String(count);
        for (const token of tokens) {
          if (token.text.toLowerCase() === guess) reveal(token);
        }
      }
      history.push({ guess, outcome });
      saveState();
    }, 50);
  });
})();
