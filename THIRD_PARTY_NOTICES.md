# Third-party notices

## MorphoBr

The `open-tables` lemmatizer data shipped in
`src/artigo_env/assets/lemmatizers/open-tables/tables.json` is derived from MorphoBr.

- Upstream: https://github.com/LR-POR/MorphoBr
- Pinned commit: `b49015cd308f128338f1aae4b49f46c887ce9608`
- Licence: Apache License 2.0
- Authors: Leonel F. de Alencar, Alexandre Rademaker, Bruno Cuconato and contributors
- Files used: `verbs/*.dict`, `nouns/*.dict`, `adjectives/*.dict`, `adverbs/adverbs.dict`
  (the SHA-256 of each file is recorded in
  `src/artigo_env/assets/lemmatizers/open-tables/MANIFEST.json`)

MorphoBr states that its resources are derived from the dictionaries of the Unitex-PB project
(NILC) and the FreeLing Portuguese data (itself derived from LABEL-LEX).

Changes: the tables are not a copy of MorphoBr. `tools/lemma_tables/build_open.py` derives
conjugation endings, verb stems, productive verb prefixes, plural and gender suffix rules and
their exception lists from the MorphoBr entries, filtered by corpus frequency; the result is
written in this project's own rule-table format.

A copy of the Apache License 2.0 is available at https://www.apache.org/licenses/LICENSE-2.0.
