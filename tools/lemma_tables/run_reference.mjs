// Usage: node run_reference.mjs <bundle.js> <forms.txt> <out.tsv>
// Runs the ORIGINAL (uninstrumented) lemmatizer over each line of forms.txt.
import { readFileSync, writeFileSync } from "node:fs";
import { evaluate, fail, loadBundle, sliceBundle } from "./_bundle.mjs";

const [bundlePath, formsPath, outPath] = process.argv.slice(2);
if (!bundlePath || !formsPath || !outPath) {
  fail("usage: node run_reference.mjs <bundle.js> <forms.txt> <out.tsv>");
}

const { text } = loadBundle(bundlePath);
const context = evaluate(sliceBundle(text).code);
if (typeof context.vn !== "function") fail("vn is not a function after evaluation");

const forms = readFileSync(formsPath, "utf8")
  .split("\n")
  .map((line) => line.replace(/\r$/, ""));

const lines = [];
for (const form of forms) {
  if (form === "") continue;
  lines.push(`${form}\t${context.vn(form)}`);
}
writeFileSync(outPath, lines.join("\n") + "\n", "utf8");
process.stdout.write(`wrote ${lines.length} lines to ${outPath}\n`);
