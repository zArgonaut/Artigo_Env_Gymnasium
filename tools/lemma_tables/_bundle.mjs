// Shared helpers: locate the lemmatizer slice in the bundle and evaluate it in an isolated vm.
// This file contains no game data; every anchor is a structural marker of the minified code.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import vm from "node:vm";

export const ANCHOR_START = 'function Ft(e,r,t){for(var n="",a={}';
export const ANCHOR_VN = "function vn(e){e=e.toLowerCase()";
export const ANCHOR_VN_END = "hn[e]=t.word,t.word}";
export const ANCHOR_STOP_END = "]),kt=";

export function fail(message) {
  process.stderr.write(`ERROR: ${message}\n`);
  process.exit(1);
}

export function mustFind(text, needle, from = 0) {
  const index = text.indexOf(needle, from);
  if (index < 0) fail(`anchor not found: ${JSON.stringify(needle)}`);
  return index;
}

export function loadBundle(path) {
  const buffer = readFileSync(path);
  const text = buffer.toString("utf8");
  const sha256 = createHash("sha256").update(buffer).digest("hex");
  return { text, sha256, size: buffer.length };
}

// Returns the slice (Ft .. end of vn) plus its offsets in characters and in bytes.
export function sliceBundle(text) {
  const start = mustFind(text, ANCHOR_START);
  const vn = mustFind(text, ANCHOR_VN, start);
  const end = mustFind(text, ANCHOR_VN_END, vn) + ANCHOR_VN_END.length;
  return {
    start,
    end,
    code: text.slice(start, end),
    byteStart: Buffer.byteLength(text.slice(0, start)),
    byteEnd: Buffer.byteLength(text.slice(0, end)),
  };
}

// Minimal stand-ins for the webpack helpers the slice uses.
function groupBy(list, fn) {
  const out = {};
  for (const item of list) {
    const key = fn(item);
    (out[key] ||= []).push(item);
  }
  return out;
}

function iterator(list) {
  let it;
  return {
    s() {
      it = list[Symbol.iterator]();
    },
    n() {
      return it.next();
    },
    e(error) {
      throw error;
    },
    f() {},
  };
}

export function evaluate(code) {
  const context = vm.createContext({});
  const prelude =
    "var R={Z:function(a){return a;}};" +
    "var zt=function(){return __groupBy;};" +
    "var wt=function(l){return __iterator(l);};";
  context.__groupBy = groupBy;
  context.__iterator = iterator;
  vm.runInContext(prelude + code, context, { filename: "lemmatizer-slice.js" });
  return context;
}
