"""Build the ``open-tables`` lemmatizer tables from open resources only.

Inputs (no game data is read here):

* MorphoBr (Apache-2.0), fetched by ``fetch_morphobr``: ``form<TAB>lemma+CLASS+TAGS`` lines in
  ``verbs/``, ``nouns/``, ``adjectives/`` and ``adverbs/``;
* a corpus form-frequency list (``form<TAB>count``), built by ``make_frequencies`` from the
  local CC-News Portuguese sample;
* the policy constants below (Portuguese grammar), chosen by measuring agreement with the game
  used as a black box (``compare.py``); no game output is ever copied into the tables.

Output: ``tables.json`` in the format read by ``artigo_env.text.lemmatizers.tables`` and a
``MANIFEST.json`` that records the inputs (MorphoBr commit, SHA-256 of every input file), the
policy and the rule counts. The build is deterministic: the same inputs give a byte-identical
``tables.json``.

Usage::

    uv run --env-file .env python -m tools.lemma_tables.build_open \
        [--morphobr DIR] [--freq FILE] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import regex

from artigo_env.core.paths import data_path

TABLE_NAME = "open-tables"
TABLE_SOURCE = (
    "MorphoBr (Apache-2.0, https://github.com/LR-POR/MorphoBr) + Portuguese grammar + "
    "CC-News PT form frequencies; built by tools/lemma_tables/build_open.py"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "artigo_env" / "assets" / "lemmatizers" / "open-tables"
NOMINAL_CLASSES = ("N", "A")
STEM = ".+?"  # prefix of the plural and gender rule sets: at least one letter is kept
_LETTERS = regex.compile(r"\p{Ll}+")

# --------------------------------------------------------------------------------------------
# Grammar: regular conjugation paradigms.
#
# Each paradigm is read from a textbook model verb in MorphoBr: its forms minus the first
# ``len(model) - cut`` letters give the endings, which are then attached to the stem
# ``infinitive[:-cut]`` of every verb of the same infinitive ending. The ``-car``, ``-gar``,
# ``-çar``, ``-cer``, ``-ger``, ``-guer``, ``-gir``, ``-guir`` and ``-cir`` paradigms carry the
# regular orthographic alternation of the stem's last consonant (``fique``, ``conheço``) and
# ``-ear`` the regular ``ei`` diphthong (``passeia``).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Paradigm:
    name: str
    ending: str  # infinitive ending that selects the paradigm
    model: str  # model verb in MorphoBr
    cut: int  # letters removed from the infinitive to get the stem
    conjugation: str  # "ar", "er" or "ir": selects the short endings


PARADIGMS: tuple[Paradigm, ...] = (
    Paradigm("car", "car", "ficar", 3, "ar"),
    Paradigm("gar", "gar", "chegar", 3, "ar"),
    Paradigm("çar", "çar", "começar", 3, "ar"),
    Paradigm("ear", "ear", "passear", 2, "ar"),
    Paradigm("ar", "ar", "tornar", 2, "ar"),
    Paradigm("guer", "guer", "erguer", 4, "er"),
    Paradigm("cer", "cer", "conhecer", 3, "er"),
    Paradigm("ger", "ger", "proteger", 3, "er"),
    Paradigm("er", "er", "vender", 2, "er"),
    Paradigm("guir", "guir", "distinguir", 4, "ir"),
    Paradigm("gir", "gir", "agir", 3, "ir"),
    Paradigm("cir", "cir", "ressarcir", 3, "ir"),
    Paradigm("ir", "ir", "partir", 2, "ir"),
)

# The short present-indicative / present-subjunctive endings that the game does not group for
# regular verbs (``torna`` and ``torno`` stay as they are). They are literal: a form is short
# when it equals ``infinitive[:-2] + ending``; forms that differ by a regular orthographic
# alternation (``fique``, ``conheço``) are not short.
SHORT_ENDINGS: Mapping[str, tuple[str, ...]] = {
    "ar": ("o", "a", "as", "e", "es"),
    "er": ("o", "e", "es", "a", "as"),
    "ir": ("o", "e", "es", "a", "as"),
}


@dataclass(frozen=True)
class Policy:
    """Every tunable choice of the build (recorded in the manifest)."""

    # A verb is included when the corpus frequency of its unambiguous verb forms (forms that
    # MorphoBr lists for no other verb and for no noun, adjective or adverb) reaches this value.
    verb_min_freq: int = 16
    irregular_min_freq: int = 16
    # Irregular-stage forms that MorphoBr also lists as a noun or adjective: "participles"
    # drops such past participles (``dado``, ``visto``), "strong-participles" only those
    # without the regular -ado/-ido shape, "nominal-and-strong-participles" every such form
    # except regular-shaped participles, "all" every such form, "none" keeps them.
    irregular_nominal: str = "participles"
    # Irregular verbs with at most this many deviating forms keep the regular behaviour for
    # their paradigm forms (short forms ungrouped); only the deviating forms are listed.
    weak_max_deviating: int = 0
    # Skip deviating forms of such verbs that only fill short slots (``sigo``, ``siga``).
    weak_skip_short_slots: bool = False
    # Irregular-verb forms that are also nouns or adjectives and whose verb readings are all in
    # the second person (rare in news: ``sede``, ``leste``) are not grouped.
    drop_nominal_second_person: bool = False
    # Gender: a whole-word rule set, built from MorphoBr, for irregular feminines whose
    # masculine no suffix pattern gives (``boa`` -> ``bom``); tried before the suffix rules.
    gender_whole_words: bool = False
    gender_whole_words_min_support: int = 3
    # Which verb a form shared by several verbs goes to: "frequency" or "alphabetical".
    priority: str = "frequency"
    # Endings never grouped besides the short ones, literal like them (``infinitive[:-2]`` +
    # ending): (conjugation, ending) pairs.
    dropped_endings: tuple[tuple[str, str], ...] = ()
    # A prefixed verb ``p + X`` lends its prefix to the irregular verb ``X`` when at least this
    # share of its forms is ``p`` + a form of ``X``.
    prefix_min_share: float = 0.8
    prefix_min_length: int = 1
    # Gender exceptions: "feminine" protects only words that MorphoBr lists as feminine and
    # never with the rule's output as lemma or masculine; "any" protects every such word.
    gender_protect: str = "feminine"
    # Plural rules: (target, replacement, protect). ``protect`` adds MorphoBr exceptions; a
    # word blocked by an exception falls through to the fallback rule (``-s`` removal), which
    # has its own MorphoBr exceptions (words it would not turn into their singular).
    plural_rules: tuple[tuple[str, str, bool], ...] = ()
    plural_fallback: tuple[str, str, bool] = ("s", "", True)
    plural_fallback_after_vowel: bool = False
    # Regular stage: verbs sharing a form with a more frequent included verb leave it to it.
    rival_exceptions: bool = False
    # List the prefixed irregular forms word by word instead of prefix rule sets.
    expand_prefixes: bool = False
    # Order of the regular paradigms: "ar-er-ir" (as listed) or "ir-er-ar".
    regular_order: str = "ar-er-ir"
    # Gender rules: (target, replacement, protect).
    gender_rules: tuple[tuple[str, str, bool], ...] = ()


# Plural rules from Portuguese grammar (first rule set; MorphoBr exceptions fall through to
# the -s fallback). "-eses" -> "-ês" was added after measurement.
PLURAL_RULES: tuple[tuple[str, str, bool], ...] = (
    ("eses", "ês", True),
    ("ões", "ão", True),
    ("ães", "ão", True),
    ("ãos", "ão", True),
    ("ais", "al", True),
    ("éis", "el", True),
    ("óis", "ol", True),
    ("ceis", "cil", True),
    ("geis", "gil", True),
    ("teis", "til", True),
    ("sseis", "ssil", True),
    ("eis", "el", True),
    ("is", "il", True),
    ("res", "r", True),
    ("zes", "z", True),
    ("ses", "s", True),
    ("ns", "m", True),
)

# Gender rules: feminine -> masculine suffix patterns read from MorphoBr pairs (each shared by
# at least 3 feminine forms), selected greedily by measured agreement with the game used as a
# black box; ``protect`` (MorphoBr exceptions for lexically feminine words) was chosen the same
# way. No generic -a -> -o rule.
GENDER_RULES: tuple[tuple[str, str, bool], ...] = (
    ("ica", "ico", False),
    ("na", "no", True),
    ("va", "vo", True),
    ("oa", "ão", False),
    ("ora", "or", True),
    ("ira", "iro", True),
    ("da", "do", True),
    ("ária", "ário", False),
    ("ima", "imo", True),
    ("osa", "oso", True),
    ("inta", "into", False),
    ("erta", "erto", False),
    ("eia", "eu", True),
    ("esa", "ês", True),
    ("ona", "ão", True),
    ("za", "z", True),
    ("ua", "u", True),
    ("ura", "uro", True),
    ("oga", "ogo", False),
    ("reta", "reto", False),
    ("iga", "igo", True),
    ("eita", "eito", True),
    ("inha", "inho", True),
    ("reia", "reu", False),
    ("ara", "aro", False),
    ("ava", "avo", False),
    ("deira", "dor", True),
    ("mã", "mão", False),
)

# Endings never grouped besides the short ones (second-person plural present/subjunctive and
# the pre-clitic first-person plural imperative), chosen by measurement.
DROPPED_ENDINGS: tuple[tuple[str, str], ...] = (
    ("ar", "ais"),
    ("ar", "eis"),
    ("ar", "emo"),
    ("er", "eis"),
    ("er", "ais"),
    ("er", "amo"),
    ("ir", "ais"),
    ("ir", "amo"),
)

DEFAULT_POLICY = Policy(
    verb_min_freq=16,
    irregular_min_freq=16,
    irregular_nominal="strong-participles",
    weak_max_deviating=12,
    weak_skip_short_slots=True,
    drop_nominal_second_person=True,
    gender_whole_words=True,
    priority="alphabetical",
    dropped_endings=DROPPED_ENDINGS,
    rival_exceptions=True,
    regular_order="ir-er-ar",
    expand_prefixes=True,
    plural_rules=PLURAL_RULES,
    gender_rules=GENDER_RULES,
)


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass
class Lexicon:
    """MorphoBr, indexed for the build."""

    verbs: dict[str, dict[str, set[str]]] = field(default_factory=dict)  # lemma -> form -> tags
    nominal: dict[str, set[tuple[str, str, str]]] = field(default_factory=dict)
    adverbs: set[str] = field(default_factory=set)
    _form_verbs: dict[str, set[str]] | None = None

    def add(self, line: str) -> None:
        line = line.rstrip("\r\n")
        if "\t" not in line:
            return
        form, analysis = line.split("\t", 1)
        parts = analysis.split("+")
        if len(parts) < 2 or not form:
            return
        lemma, cls, tags = parts[0], parts[1], "+".join(parts[2:])
        if cls == "V":
            self.verbs.setdefault(lemma, {}).setdefault(form, set()).add(tags)
        elif cls in NOMINAL_CLASSES:
            self.nominal.setdefault(form, set()).add((lemma, cls, tags))
        elif cls == "ADV":
            self.adverbs.add(form)
        self._form_verbs = None

    def form_verbs(self) -> dict[str, set[str]]:
        if self._form_verbs is None:
            index: dict[str, set[str]] = {}
            for lemma, forms in self.verbs.items():
                for form in forms:
                    index.setdefault(form, set()).add(lemma)
            self._form_verbs = index
        return self._form_verbs


def read_morphobr(root: Path) -> Lexicon:
    lexicon = Lexicon()
    for directory in ("verbs", "nouns", "adjectives", "adverbs"):
        for path in sorted((root / directory).glob("*.dict")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    lexicon.add(line)
    return lexicon


def read_frequencies(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            form, _, count = line.rstrip("\n").partition("\t")
            if form and count:
                counts[form] = counts.get(form, 0) + int(count)
    return counts


# --------------------------------------------------------------------------------------------
# Verbs
# --------------------------------------------------------------------------------------------


def paradigm_endings(lexicon: Lexicon) -> dict[str, tuple[str, ...]]:
    """Endings of every paradigm, read from its model verb (sorted, deterministic)."""
    out: dict[str, tuple[str, ...]] = {}
    for paradigm in PARADIGMS:
        forms = lexicon.verbs.get(paradigm.model)
        if not forms:
            continue  # without its model verb the paradigm is not used
        stem = paradigm.model[: -paradigm.cut]
        endings = sorted(form[len(stem) :] for form in forms if form.startswith(stem))
        out[paradigm.name] = tuple(endings)
    return out


def paradigm_for(infinitive: str, available: Iterable[str] | None = None) -> Paradigm | None:
    names = None if available is None else set(available)
    for paradigm in PARADIGMS:
        if names is not None and paradigm.name not in names:
            continue
        if infinitive.endswith(paradigm.ending) and len(infinitive) > paradigm.cut:
            return paradigm
    return None


def short_endings_for(
    infinitive: str, paradigm: Paradigm, dropped: Iterable[tuple[str, str]] = ()
) -> set[str]:
    """Paradigm endings (relative to the paradigm stem) that make a literal short form.

    ``dropped`` adds further literal endings, as ``(conjugation, ending)`` pairs.
    """
    stem = infinitive[: -paradigm.cut]
    base = infinitive[:-2]
    literal = list(SHORT_ENDINGS[paradigm.conjugation])
    literal += [ending for conjugation, ending in dropped if conjugation == paradigm.conjugation]
    return {(base + ending)[len(stem) :] for ending in literal}


def verb_frequencies(lexicon: Lexicon, freq: Mapping[str, int]) -> dict[str, int]:
    """Corpus frequency of each verb over its unambiguous verb forms."""
    form_verbs = lexicon.form_verbs()
    out: dict[str, int] = {}
    for lemma, forms in lexicon.verbs.items():
        total = 0
        for form in forms:
            unambiguous = len(form_verbs[form]) == 1
            if unambiguous and form not in lexicon.nominal and form not in lexicon.adverbs:
                total += freq.get(form, 0)
        out[lemma] = total
    return out


def _is_participle(tags: Iterable[str]) -> bool:
    return any(tag.startswith("PTPST") for tag in tags)


def _alternation(words: Iterable[str]) -> str:
    return "|".join(regex.escape(word) for word in words)


@dataclass
class VerbPlan:
    regular: dict[str, list[tuple[str, str, str]]]  # paradigm -> rules (stem, inf, exceptions)
    regular_suffix: dict[str, str]
    irregular: list[tuple[str, str, str]]  # (form, infinitive, "")
    prefixed: list[tuple[str, list[tuple[str, str, str]]]]  # (prefix regex, rules)
    stats: dict[str, int]


def deviating_forms(
    infinitive: str, forms: Mapping[str, set[str]], paradigm: Paradigm, endings: Sequence[str]
) -> list[str]:
    """Forms outside the paradigm, ignoring accent variants and extra past participles."""
    stem = infinitive[: -paradigm.cut]
    allowed = {stem + ending for ending in endings}
    plain = {_deaccent(form) for form in allowed}
    out: list[str] = []
    for form in sorted(forms):
        if form in allowed or _deaccent(form) in plain:
            continue
        if all(tag.startswith("PTPST") for tag in forms[form]):
            continue
        out.append(form)
    return out


def verb_kind(
    infinitive: str,
    forms: Mapping[str, set[str]],
    endings: Mapping[str, tuple[str, ...]],
    weak_max: int = 0,
) -> tuple[str, Paradigm | None]:
    """``("regular" | "near" | "weak" | "irregular", paradigm)``.

    ``regular``: every MorphoBr form is stem + paradigm ending. ``near``: the forms outside the
    paradigm are only accent variants of paradigm forms (``apóio``, ``saúdo``) or extra past
    participles (``morto`` for ``matar``); such verbs behave like regular ones. ``weak``: at
    most ``weak_max`` other forms deviate (stem alternation: ``sigo``/``siga`` for
    ``seguir``); the paradigm part behaves like a regular verb and only the deviating forms
    are listed. ``irregular``: every form is listed.
    """
    paradigm = paradigm_for(infinitive, endings)
    if paradigm is None:
        return "irregular", None
    stem = infinitive[: -paradigm.cut]
    allowed = {stem + ending for ending in endings[paradigm.name]}
    if all(form in allowed for form in forms):
        return "regular", paradigm
    deviating = deviating_forms(infinitive, forms, paradigm, endings[paradigm.name])
    if not deviating:
        return "near", paradigm
    if len(deviating) <= weak_max:
        return "weak", paradigm
    return "irregular", paradigm


SHORT_SLOTS = frozenset(
    ("PRS+1+SG", "PRS+2+SG", "PRS+3+SG", "SBJR+1+SG", "SBJR+2+SG", "SBJR+3+SG")
    + ("IMP+2+SG", "IMP+3+SG")
)


def plan_verbs(lexicon: Lexicon, freq: Mapping[str, int], policy: Policy) -> VerbPlan:
    endings = paradigm_endings(lexicon)
    vfreq = verb_frequencies(lexicon, freq)
    form_verbs = lexicon.form_verbs()
    if policy.priority == "alphabetical":
        ordered = sorted(lexicon.verbs)
    else:  # more frequent first; ties broken alphabetically
        ordered = sorted(lexicon.verbs, key=lambda lemma: (-vfreq[lemma], lemma))
    dropped = policy.dropped_endings
    regular: dict[str, list[tuple[str, str, str]]] = {p.name: [] for p in PARADIGMS}
    present = [p for p in PARADIGMS if p.name in endings]
    irregular_verbs: list[str] = []
    kinds: dict[str, str] = {}
    listed: list[tuple[str, str]] = []  # (form, infinitive) for the irregular stage
    for lemma in ordered:
        if not _LETTERS.fullmatch(lemma):
            continue
        forms = lexicon.verbs[lemma]
        kind, paradigm = verb_kind(lemma, forms, endings, policy.weak_max_deviating)
        if kind == "irregular":
            if vfreq[lemma] >= policy.irregular_min_freq:
                irregular_verbs.append(lemma)
                kinds[lemma] = kind
            continue
        assert paradigm is not None
        if vfreq[lemma] < policy.verb_min_freq:
            continue
        kinds[lemma] = kind
        stem = lemma[: -paradigm.cut]
        short = short_endings_for(lemma, paradigm, dropped)
        generated = {stem + e for e in endings[paradigm.name] if e not in short}
        # Forms the paradigm generates that MorphoBr knows only as a noun or adjective
        # (never as a form of this verb) are exceptions.
        blocked_set = {f for f in generated if f not in forms and f in lexicon.nominal}
        if policy.rival_exceptions:
            # A form that a more frequent included verb also has belongs to that verb, which
            # may leave it ungrouped (``segue``: short form of ``seguir``, not of ``segar``).
            for f in generated:
                for other in form_verbs.get(f, ()):
                    stronger = vfreq[other] > vfreq[lemma]
                    if other != lemma and stronger and vfreq[other] >= policy.verb_min_freq:
                        blocked_set.add(f)
        if policy.drop_nominal_second_person:
            for f in generated:
                tags = forms.get(f)
                if tags and f in lexicon.nominal and all("+2+" in t for t in tags):
                    blocked_set.add(f)
        blocked = sorted(blocked_set)
        regular[paradigm.name].append((stem, lemma, _alternation(blocked)))
        if kind == "regular":
            continue
        allowed = {stem + e for e in endings[paradigm.name]}
        allowed_plain = {_deaccent(f) for f in allowed}
        short_plain = {_deaccent(lemma[:-2] + e) for e in SHORT_ENDINGS[paradigm.conjugation]}
        for form, tags in sorted(forms.items()):
            if form in allowed or _deaccent(form) in short_plain:
                continue
            if _deaccent(form) not in allowed_plain and all(t.startswith("PTPST") for t in tags):
                continue  # extra (strong) participles are not grouped
            deviating = _deaccent(form) not in allowed_plain  # a deviating form of a weak verb
            if deviating and policy.weak_skip_short_slots and tags <= SHORT_SLOTS:
                continue
            listed.append((form, lemma))

    included = set(kinds)
    for lemma in irregular_verbs:
        for form, tags in sorted(lexicon.verbs[lemma].items()):
            if form in lexicon.nominal and _drop_nominal(lemma, form, tags, policy):
                continue
            # A form that a more frequent non-irregular verb also has is left to that verb
            # (``começa`` is a short form of ``começar``, not a form of ``comedir``).
            rivals = [
                v
                for v in form_verbs[form]
                if v != lemma and v in included and kinds[v] != "irregular"
            ]
            if any(vfreq[v] > vfreq[lemma] for v in rivals):
                continue
            listed.append((form, lemma))

    regular_suffix: dict[str, str] = {}
    for paradigm in present:
        # The suffix alternation is shared by every verb of the paradigm; the short endings,
        # relative to the paradigm stem, are the same for all of them (read from the model).
        model_short = short_endings_for(paradigm.model, paradigm, dropped)
        kept = [e for e in endings[paradigm.name] if e not in model_short]
        kept.sort(key=lambda e: (-len(e), e))
        regular_suffix[paradigm.name] = "(?:" + _alternation(kept) + ")"

    # Each listed form goes to the first verb in priority order, except that the infinitive
    # of an included verb always stays with that verb.
    assigned: dict[str, str] = {}
    for form, lemma in listed:
        if _LETTERS.fullmatch(form):
            assigned.setdefault(form, lemma)
    for lemma in sorted(included):
        if lemma in assigned:
            if kinds[lemma] == "irregular":
                assigned[lemma] = lemma
            else:
                del assigned[lemma]
    irregular = [(form, assigned[form], "") for form in sorted(assigned)]

    prefixed = plan_prefixes(lexicon, irregular_verbs, policy)
    stats = {
        "verbs_regular": sum(1 for k in kinds.values() if k == "regular"),
        "verbs_near_regular": sum(1 for k in kinds.values() if k == "near"),
        "verbs_weak_irregular": sum(1 for k in kinds.values() if k == "weak"),
        "verbs_irregular": len(irregular_verbs),
        "irregular_forms": len(irregular),
    }
    return VerbPlan(regular, regular_suffix, irregular, prefixed, stats)


def _drop_nominal(infinitive: str, form: str, tags: set[str], policy: Policy) -> bool:
    """Whether an irregular-verb form that is also a noun or adjective stays ungrouped."""
    if policy.drop_nominal_second_person and tags and all("+2+" in t for t in tags):
        return True
    mode = policy.irregular_nominal
    if mode == "none":
        return False
    if mode == "all":
        return True
    if not _is_participle(tags):
        return mode == "nominal-and-strong-participles"
    if mode in ("strong-participles", "nominal-and-strong-participles"):
        # Participles with the regular shape (stem + -ado/-ido/-ído) stay grouped.
        stem = infinitive[:-2]
        regular_shape = (stem + "ad", stem + "id", stem + "íd", _deaccent(stem) + "íd")
        return not form.startswith(regular_shape)
    return True  # "participles"


def _deaccent(text: str) -> str:
    return text.translate(str.maketrans("áâãàéêíóôõúç", "aaaaeeiooouc"))


def plan_prefixes(
    lexicon: Lexicon, irregular_verbs: Sequence[str], policy: Policy
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Productive prefixes of irregular verbs, validated against MorphoBr.

    For an irregular verb ``X`` and a MorphoBr verb ``Y = p + X`` (or ``p`` + ``X`` without
    accents, as in ``propor``/``pôr``), ``p`` is a prefix of ``X`` when most forms of ``Y`` are
    ``p`` + a form of ``X``. The rule targets are the forms of every such ``Y`` with ``p``
    removed, so that each target comes from a real prefixed form (``detém`` gives ``tém``).
    Verbs that share the same prefix set share one rule set whose prefix is mandatory.
    """
    form_verbs = lexicon.form_verbs()
    groups: dict[tuple[str, ...], dict[str, str]] = {}
    for base in irregular_verbs:
        base_forms = set(lexicon.verbs[base])
        keys = {base, _deaccent(base)}
        prefixes: dict[str, set[str]] = {}
        for lemma in lexicon.verbs:
            if lemma == base:
                continue
            for key in keys:
                if not lemma.endswith(key) or len(lemma) - len(key) < policy.prefix_min_length:
                    continue
                prefix = lemma[: -len(key)]
                if not _LETTERS.fullmatch(prefix):
                    continue
                forms = lexicon.verbs[lemma]
                stripped = {f[len(prefix) :] for f in forms if f.startswith(prefix)}
                share = len(stripped & base_forms) / max(1, len(forms))
                if share < policy.prefix_min_share:
                    continue
                for form, tags in forms.items():
                    if not form.startswith(prefix):
                        continue
                    if form_verbs[form] != {lemma}:
                        continue  # shared with another verb: leave it to that verb
                    if form in lexicon.nominal and _drop_nominal(lemma, form, tags, policy):
                        continue
                    prefixes.setdefault(prefix, set()).add(form[len(prefix) :])
        if not prefixes:
            continue
        key_set = tuple(sorted(prefixes, key=lambda p: (-len(p), p)))
        targets = groups.setdefault(key_set, {})
        endings_of_base = tuple(sorted(keys))
        for prefix in key_set:
            for form in prefixes[prefix]:
                if not form or not _LETTERS.fullmatch(form):
                    continue
                # Every prefix of the set applies to every target: drop a target that some
                # prefix turns into a word MorphoBr knows otherwise (``a`` + ``valia``).
                clash = False
                for other in key_set:
                    word = other + form
                    owners = form_verbs.get(word, set())
                    foreign = any(not owner.endswith(endings_of_base) for owner in owners)
                    dropped_nominal = word in lexicon.nominal and (
                        not owners
                        or any(
                            _drop_nominal(o, word, lexicon.verbs[o][word], policy) for o in owners
                        )
                    )
                    if foreign or dropped_nominal:
                        clash = True
                if not clash:
                    targets.setdefault(form, base)
    out: list[tuple[str, list[tuple[str, str, str]]]] = []
    for key_set in sorted(groups, key=lambda k: (-len(k), k)):
        rules = [(form, base, "") for form, base in sorted(groups[key_set].items())]
        out.append(("(?:" + _alternation(key_set) + ")", rules))
    return out


# --------------------------------------------------------------------------------------------
# Plural and gender
# --------------------------------------------------------------------------------------------


def _gender(tags: str) -> str:
    parts = tags.split("+")
    return "F" if "F" in parts else "M" if "M" in parts else ""


def singular_index(lexicon: Lexicon) -> dict[tuple[str, str, str, str], set[str]]:
    """(lemma, class, gender, degree) -> singular forms."""
    index: dict[tuple[str, str, str, str], set[str]] = {}
    for form, analyses in lexicon.nominal.items():
        for lemma, cls, tags in analyses:
            parts = tags.split("+")
            if "PL" in parts or not _plain(tags):
                continue
            index.setdefault((lemma, cls, _gender(tags), ""), set()).add(form)
    return index


_DEGREES = ("DIM", "AUG", "SUPER")


def _plain(tags: str) -> bool:
    """An analysis without derivational degree (diminutive, augmentative, superlative)."""
    return not any(part in _DEGREES for part in tags.split("+"))


def winning_target(form: str, targets: Sequence[str]) -> str | None:
    """The target a ``.+?``-prefixed suffix rule set picks for ``form``: the longest suffix."""
    best: str | None = None
    for target in targets:
        matches = form.endswith(target) and len(form) > len(target)
        if matches and (best is None or len(target) > len(best)):
            best = target
    return best


def _supports_plural(
    output: str,
    analyses: Iterable[tuple[str, str, str]],
    singulars: Mapping[tuple[str, str, str, str], set[str]],
) -> bool:
    for lemma, cls, tags in analyses:
        if "PL" not in tags.split("+"):
            continue
        if output == lemma or output in singulars.get((lemma, cls, _gender(tags), ""), set()):
            return True
    return False


def plural_exceptions(
    lexicon: Lexicon,
    rules: Sequence[tuple[str, str, bool]],
    fallback: tuple[str, str, bool],
    singulars: Mapping[tuple[str, str, str, str], set[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """MorphoBr exceptions of the plural rules and of the fallback rule.

    A nominal form is an exception of the rule that captures it when the rule's output is not
    its singular. Only analyses without a derivational degree count (the game treats
    diminutives and augmentatives through the suffix rules). Forms blocked in the first rule
    set reach the fallback, so they are checked against it too.
    """
    replacement = {target: repl for target, repl, _ in rules}
    protected = {target for target, _, protect in rules if protect}
    targets = [target for target, _, _ in rules]
    out: dict[str, list[str]] = {target: [] for target in targets}
    last: list[str] = []
    f_target, f_replacement, _ = fallback
    for form, analyses in lexicon.nominal.items():
        if not _LETTERS.fullmatch(form):
            continue
        plain = [a for a in analyses if _plain(a[2])]
        if not plain:
            continue
        target = winning_target(form, targets)
        if target is not None:
            output = form[: len(form) - len(target)] + replacement[target]
            if _supports_plural(output, plain, singulars):
                continue
            out[target].append(form)
            if target not in protected:
                continue
        if form.endswith(f_target) and len(form) > len(f_target):
            output = form[: len(form) - len(f_target)] + f_replacement
            if not _supports_plural(output, plain, singulars):
                last.append(form)
    return {target: sorted(forms) for target, forms in out.items()}, sorted(last)


def gender_exceptions(
    lexicon: Lexicon, rules: Sequence[tuple[str, str, bool]], mode: str
) -> dict[str, list[str]]:
    """Per target: singular nominal forms it captures that never have the output as lemma."""
    masculine: dict[tuple[str, str], set[str]] = {}
    for form, analyses in lexicon.nominal.items():
        for lemma, cls, tags in analyses:
            parts = tags.split("+")
            if "M" in parts and "PL" not in parts and _plain(tags):
                masculine.setdefault((lemma, cls), set()).add(form)
    replacement = {target: repl for target, repl, _ in rules}
    targets = [target for target, _, _ in rules]
    out: dict[str, list[str]] = {target: [] for target in targets}
    for form, analyses in lexicon.nominal.items():
        if not _LETTERS.fullmatch(form):
            continue
        plain = [a for a in analyses if _plain(a[2])]
        if not plain:
            continue
        target = winning_target(form, targets)
        if target is None:
            continue
        output = form[: len(form) - len(target)] + replacement[target]
        supported = any(
            output == lemma or output in masculine.get((lemma, cls), set())
            for lemma, cls, _ in plain
        )
        if supported:
            continue
        if mode == "feminine" and not any(_gender(tags) == "F" for _, _, tags in plain):
            continue
        out[target].append(form)
    return {target: sorted(forms) for target, forms in out.items()}


def feminine_pairs(lexicon: Lexicon) -> dict[str, set[str]]:
    """Plain feminine singular nominal forms -> masculine singular forms of the same lemma."""
    masculine: dict[tuple[str, str], set[str]] = {}
    for form, analyses in lexicon.nominal.items():
        for lemma, cls, tags in analyses:
            parts = tags.split("+")
            if "M" in parts and "SG" in parts and _plain(tags):
                masculine.setdefault((lemma, cls), set()).add(form)
    pairs: dict[str, set[str]] = {}
    for form, analyses in lexicon.nominal.items():
        if not _LETTERS.fullmatch(form):
            continue
        for lemma, cls, tags in analyses:
            parts = tags.split("+")
            if "F" in parts and "SG" in parts and _plain(tags):
                for male in masculine.get((lemma, cls), set()):
                    if male != form:
                        pairs.setdefault(form, set()).add(male)
    return pairs


def suffix_patterns(
    pairs: Mapping[str, set[str]], max_context: int = 3
) -> dict[tuple[str, str], int]:
    """Support (number of feminine forms) of every feminine -> masculine suffix pattern."""
    support: dict[tuple[str, str], set[str]] = {}
    for female, males in pairs.items():
        for male in males:
            k = 0
            while k < min(len(female), len(male)) and female[k] == male[k]:
                k += 1
            for context in range(max_context + 1):
                if k - context < 1:
                    break
                support.setdefault((female[k - context :], male[k - context :]), set()).add(female)
    return {pattern: len(forms) for pattern, forms in support.items()}


def irregular_feminines(lexicon: Lexicon, min_support: int) -> list[tuple[str, str]]:
    """Feminine forms whose masculine follows no suffix pattern shared by ``min_support``
    feminine forms (``boa`` -> ``bom``, ``má`` -> ``mau``); one masculine per form."""
    pairs = feminine_pairs(lexicon)
    support = suffix_patterns(pairs)
    out: list[tuple[str, str]] = []
    for female in sorted(pairs):
        males = sorted(pairs[female])
        if len(males) != 1:
            continue
        male = males[0]
        k = 0
        while k < min(len(female), len(male)) and female[k] == male[k]:
            k += 1
        if support.get((female[k:], male[k:]), 0) < min_support:
            out.append((female, male))
    return out


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------


def group_forms(rules: Sequence[Sequence[str]]) -> list[list[str]]:
    """Merge whole-word rules with the same lemma and first letter into one target.

    ``tem -> ter`` and ``tinha -> ter`` become ``t(?:em|inha) -> ter``: the same behaviour with
    far fewer alternatives, which makes the rule set much faster to apply.
    """
    merged: dict[tuple[str, str], list[str]] = {}
    for form, lemma, exceptions in rules:
        if exceptions:
            raise ValueError("whole-word rules carry no exceptions")
        merged.setdefault((form[0], lemma), []).append(form[1:])
    out: list[list[str]] = []
    for (first, lemma), tails in sorted(merged.items()):
        tails = sorted(set(tails), key=lambda t: (-len(t), t))
        if len(tails) == 1:
            target = regex.escape(first + tails[0])
        else:
            target = regex.escape(first) + "(?:" + _alternation(tails) + ")"
        out.append([target, lemma, ""])
    return out


def build_tables(
    lexicon: Lexicon, freq: Mapping[str, int], policy: Policy = DEFAULT_POLICY
) -> tuple[dict[str, object], dict[str, int]]:
    plan = plan_verbs(lexicon, freq, policy)
    irregular_rules = [list(r) for r in plan.irregular]
    irregular_members: list[dict[str, object]] = []
    if policy.expand_prefixes:
        # Same behaviour as the prefixed rule sets, listed word by word: one indexed rule set
        # instead of dozens is much faster to apply.
        listed = {rule[0] for rule in irregular_rules}
        extra: dict[str, str] = {}
        for prefix_regex, rules in plan.prefixed:
            prefixes = prefix_regex[len("(?:") : -1].split("|")
            for prefix in prefixes:
                for target, base, _ in rules:
                    word = prefix + target
                    if word not in listed:
                        extra.setdefault(word, prefix + base)
        irregular_rules = sorted(irregular_rules + [[w, m, ""] for w, m in extra.items()])
    irregular_members.append(
        {"type": "indexed", "prefix": "", "suffix": "", "rules": group_forms(irregular_rules)}
    )
    if not policy.expand_prefixes:
        for prefix, rules in plan.prefixed:
            listed_rules = [list(r) for r in rules]
            irregular_members.append(
                {"type": "indexed", "prefix": prefix, "suffix": "", "rules": listed_rules}
            )
    regular_members: list[dict[str, object]] = []
    paradigms = list(PARADIGMS)
    if policy.regular_order == "ir-er-ar":
        rank = {"ir": 0, "er": 1, "ar": 2}
        paradigms.sort(key=lambda p: rank[p.conjugation])
    for paradigm in paradigms:
        rules = plan.regular[paradigm.name]
        if rules:
            regular_members.append(
                {
                    "type": "indexed",
                    "prefix": "",
                    "suffix": plan.regular_suffix[paradigm.name],
                    "rules": [list(r) for r in rules],
                }
            )
    singulars = singular_index(lexicon)
    plural_blocked, fallback_blocked = plural_exceptions(
        lexicon, policy.plural_rules, policy.plural_fallback, singulars
    )
    plural_rules = [
        [target, replacement, _alternation(plural_blocked[target]) if protect else ""]
        for target, replacement, protect in policy.plural_rules
    ]
    f_target, f_replacement, f_protect = policy.plural_fallback
    fallback_rule = [f_target, f_replacement, _alternation(fallback_blocked) if f_protect else ""]
    if policy.plural_fallback_after_vowel:
        # Portuguese plurals in -s follow a vowel; "reuters" or "fgts" are left alone.
        fallback_rule[0] = "(?<=[aeiouáéíóúâêôãõ])" + f_target
    gender_blocked = gender_exceptions(lexicon, policy.gender_rules, policy.gender_protect)
    gender_rules = [
        [target, replacement, _alternation(gender_blocked[target]) if protect else ""]
        for target, replacement, protect in policy.gender_rules
    ]
    gender_members: list[dict[str, object]] = []
    whole: list[tuple[str, str]] = []
    if policy.gender_whole_words:
        whole = irregular_feminines(lexicon, policy.gender_whole_words_min_support)
        gender_members.append(
            {"type": "indexed", "prefix": "", "suffix": "", "rules": [[f, m, ""] for f, m in whole]}
        )
    gender_members.append({"type": "suffix", "prefix": STEM, "suffix": "", "rules": gender_rules})
    tables: dict[str, object] = {
        "format_version": 1,
        "name": TABLE_NAME,
        "source": TABLE_SOURCE,
        "stages": {
            "irregular": {"type": "first", "members": irregular_members},
            "regular": {"type": "first", "members": regular_members},
            "plural": {
                "type": "first",
                "members": [
                    {"type": "suffix", "prefix": STEM, "suffix": "", "rules": plural_rules},
                    {"type": "suffix", "prefix": STEM, "suffix": "", "rules": [fallback_rule]},
                ],
            },
            "gender": {"type": "first", "members": gender_members},
        },
    }
    stats = dict(plan.stats)
    stats["prefix_rule_sets"] = len(plan.prefixed)
    stats["prefixed_targets"] = sum(len(rules) for _, rules in plan.prefixed)
    stats["plural_rules"] = len(plural_rules)
    stats["plural_exceptions"] = sum(
        r[2].count("|") + 1 for r in [*plural_rules, fallback_rule] if r[2]
    )
    stats["gender_rules"] = len(gender_rules)
    stats["gender_whole_words"] = len(whole)
    stats["gender_exceptions"] = sum(r[2].count("|") + 1 for r in gender_rules if r[2])
    return tables, stats


def dump_tables(tables: Mapping[str, object]) -> str:
    """Stable JSON text: one rule per line, so diffs and the forbidden-tables check stay sane."""
    text = json.dumps(tables, ensure_ascii=False, indent=1, sort_keys=False)
    # Collapse each [target, replacement, exceptions] triple onto one line.
    string = r'"(?:[^"\\]|\\.)*"'
    triple = regex.compile(rf"\[\n\s*({string}),\n\s*({string}),\n\s*({string})\n\s*\]")
    return triple.sub(r"[\1, \2, \3]", text) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--morphobr", type=Path, default=None)
    parser.add_argument("--freq", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    morphobr = args.morphobr or data_path("raw", "morphobr")
    freq_path = args.freq or data_path("measurements", "cc-news-form-freq.tsv")
    lexicon = read_morphobr(morphobr)
    freq = read_frequencies(freq_path)
    tables, stats = build_tables(lexicon, freq, DEFAULT_POLICY)
    text = dump_tables(tables)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables.json").write_text(text, encoding="utf-8", newline="\n")
    upstream = json.loads((morphobr / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest = {
        "name": TABLE_NAME,
        "tables_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "inputs": {
            "morphobr": {
                "source": upstream["source"],
                "license": upstream["license"],
                "commit": upstream["commit"],
                "files": upstream["files"],
            },
            "frequencies": {
                "description": "CC-News PT sample, tokenized with artigo_env.text.tokenize, "
                "lower-cased (tools/lemma_tables/make_frequencies.py)",
                "sha256": sha256_file(freq_path),
                "forms": len(freq),
            },
        },
        "policy": asdict(DEFAULT_POLICY),
        "paradigms": [asdict(p) for p in PARADIGMS],
        "short_endings": dict(SHORT_ENDINGS),
        "stats": stats,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
