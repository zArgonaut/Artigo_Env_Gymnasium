# lemma_tables

Local-only tools that turn a saved copy of the game's public client bundle into a
**measuring stick** for our own lemmatizers (`reference-local`).

## Boundary rule

Nothing extracted from the bundle (rule tables, exception lists, stopword list, the code itself)
may be written inside this repository. Outputs go only to
`<ARTIGO_DATA_ROOT>/reference/lemmatizer/`. The repository holds only these scripts (no game data)
and tests. The pre-commit hook `tools.repo.check_forbidden_tables` blocks accidental commits using
`signatures.txt` produced by the extractor. Nothing here contacts the network.

## Run

```bash
BUNDLE=D:/ArtigoEnv/data/reference/bundle/main.be54b82b.js
OUT=D:/ArtigoEnv/data/reference/lemmatizer

# 1. tables.json, stopwords.txt, signatures.txt, meta.json
node tools/lemma_tables/extract_reference.mjs $BUNDLE $OUT

# 2. run the ORIGINAL lemmatizer over a list of forms (one per line)
node tools/lemma_tables/run_reference.mjs $BUNDLE $OUT/reference_forms_input.txt $OUT/reference_forms.tsv
```

The extractor fails with a non-zero exit code and a clear message if any anchor is missing, which
signals that the bundle changed.

## Files

- `extract_reference.mjs`: slices the bundle from `function Ft(` to the end of `function vn`,
  instruments the three constructors so each instance records its arguments, evaluates the slice in
  `node:vm` with minimal stand-ins for the webpack helpers, and writes the artifacts.
- `run_reference.mjs`: evaluates the uninstrumented slice and writes `form<TAB>lemma` lines.
- `_bundle.mjs`: shared slicing and evaluation helpers.

## Output format (`tables.json`)

`{"format_version": 1, "name": "reference-local", "source": "...", "stages": {...}}` with stages
`irregular`, `regular`, `plural`, `gender`. A node is `{"type": "first", "members": [...]}`,
`{"type": "suffix", "prefix", "suffix", "rules"}` or `{"type": "indexed", ...}`; a rule is
`[target, replacement, exceptions]` and order is preserved.
The `local` test `tests/fidelity/reference/test_reference_port.py` checks 100% agreement between
`RuleLemmatizer` and `reference_forms.tsv`.
