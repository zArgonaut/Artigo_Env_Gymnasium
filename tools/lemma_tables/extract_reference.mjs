// Usage: node extract_reference.mjs <bundle.js> <out-dir>
// Instruments the three rule constructors, evaluates the lemmatizer slice and dumps its tables.
// Output stays in <out-dir> (outside the repository); never commit it.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";
import {
  ANCHOR_STOP_END,
  evaluate,
  fail,
  loadBundle,
  sliceBundle,
} from "./_bundle.mjs";

const [bundlePath, outDir] = process.argv.slice(2);
if (!bundlePath || !outDir) fail("usage: node extract_reference.mjs <bundle.js> <out-dir>");

const { text, sha256, size } = loadBundle(bundlePath);
const slice = sliceBundle(text);

function instrument(code) {
  const edits = [
    ["function Ft(e,r,t){", 'function Ft(e,r,t){this.__ctor=["suffix",e,r,t];'],
    ["function At(e,r,t){", 'function At(e,r,t){this.__ctor=["indexed",e,r,t];'],
    ["function Ot(e){", 'function Ot(e){this.__ctor=["first",e];'],
  ];
  let out = code;
  for (const [from, to] of edits) {
    if (out.split(from).length !== 2) fail(`expected exactly one ${from} in the slice`);
    out = out.replace(from, () => to);
  }
  return out;
}

const context = evaluate(instrument(slice.code));

const counts = { stages: {}, nodes: { first: 0, suffix: 0, indexed: 0 }, exceptions: 0 };
const exceptionStrings = [];

function ruleList(rules, where) {
  if (!Array.isArray(rules)) fail(`rules is not an array at ${where}`);
  return rules.map((rule, i) => {
    if (!Array.isArray(rule) || typeof rule[0] !== "string" || typeof rule[1] !== "string") {
      fail(`malformed rule ${i} at ${where}`);
    }
    const exceptions = rule[2] ? String(rule[2]) : "";
    if (exceptions) {
      counts.exceptions += 1;
      exceptionStrings.push(exceptions);
    }
    return [rule[0], rule[1], exceptions];
  });
}

function toNode(instance, where) {
  const ctor = instance && instance.__ctor;
  if (!ctor) fail(`object without constructor record at ${where}`);
  const [kind, a, b, c] = ctor;
  counts.nodes[kind] += 1;
  if (kind === "first") {
    return { type: "first", members: a.map((m, i) => toNode(m, `${where}.${i}`)) };
  }
  if (typeof a !== "string" || typeof b !== "string") fail(`non-string prefix/suffix at ${where}`);
  return { type: kind, prefix: a, suffix: b, rules: ruleList(c, where) };
}

function countRules(node) {
  if (node.type === "first") return node.members.reduce((s, m) => s + countRules(m), 0);
  return node.rules.length;
}

const sources = { irregular: "Rt", regular: "mn", plural: "Zt", gender: "Pt" };
const stages = {};
for (const [stage, variable] of Object.entries(sources)) {
  if (!context[variable]) fail(`variable ${variable} (${stage}) not found after evaluation`);
  stages[stage] = toNode(context[variable], stage);
  counts.stages[stage] = countRules(stages[stage]);
}

// Stopwords: the array literal passed to the IIFE that ends right before the lodash helper.
const stopEnd = text.lastIndexOf(ANCHOR_STOP_END, slice.start);
if (stopEnd < 0) fail("stopword Set end anchor not found");
const setStart = text.lastIndexOf("}([", stopEnd);
if (setStart < 0 || text.slice(setStart + 2, stopEnd).includes("]")) {
  fail("stopword array literal not found");
}
const literal = text.slice(setStart + 2, stopEnd + 1);
let rawStopwords;
try {
  rawStopwords = vm.runInNewContext(literal);
} catch (error) {
  fail(`stopword literal does not evaluate: ${error.message.slice(0, 120)}`);
}
if (!Array.isArray(rawStopwords) || !rawStopwords.every((w) => typeof w === "string")) {
  fail("stopword literal is not an array of strings");
}
const stopwords = [...new Set(rawStopwords)];

// Signatures: 40 distinctive runs of exception alternatives (>= 60 chars), evenly spread.
function chunks(exceptions) {
  const out = [];
  let current = [];
  for (const alt of exceptions.split("|")) {
    current.push(alt);
    if (current.join("|").length >= 60) {
      out.push(current.join("|"));
      current = [];
    }
  }
  return out;
}
const candidates = [...new Set(exceptionStrings.flatMap(chunks))].filter(
  (s) => s.length >= 60 && ![...s].some((c) => c.charCodeAt(0) < 32),
);
if (candidates.length < 40) fail(`only ${candidates.length} signature candidates >= 60 chars`);
const signatures = [];
for (let i = 0; i < 40; i++) signatures.push(candidates[Math.floor((i * candidates.length) / 40)]);

mkdirSync(outDir, { recursive: true });
const tables = {
  format_version: 1,
  name: "reference-local",
  source: `artigo.app bundle ${sha256.slice(0, 16)}`,
  stages,
};
const meta = {
  bundle_sha256: sha256,
  bundle_bytes: size,
  slice: {
    start_char: slice.start,
    end_char: slice.end,
    start_byte: slice.byteStart,
    end_byte: slice.byteEnd,
  },
  rule_counts: counts.stages,
  node_counts: counts.nodes,
  rules_with_exceptions: counts.exceptions,
  stopwords_raw: rawStopwords.length,
  stopwords_unique: stopwords.length,
};
writeFileSync(join(outDir, "tables.json"), JSON.stringify(tables) + "\n", "utf8");
writeFileSync(join(outDir, "stopwords.txt"), stopwords.join("\n") + "\n", "utf8");
writeFileSync(join(outDir, "signatures.txt"), signatures.join("\n") + "\n", "utf8");
writeFileSync(join(outDir, "meta.json"), JSON.stringify(meta, null, 2) + "\n", "utf8");
process.stdout.write(`extracted to ${outDir}: ${JSON.stringify(meta.rule_counts)}\n`);
