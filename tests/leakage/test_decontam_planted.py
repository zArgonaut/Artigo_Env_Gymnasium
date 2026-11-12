"""Planted-case suite of the decontamination barriers (SDD 06 §7.6).

Port of the reference script ``teste_descontaminacao.py`` (2026-10-29): synthetic W articles and
training documents with planted contamination of 11 kinds. The texts are built with the same
``random.Random(seed)`` draws as the script, so seed 42 reproduces its default run. The flow is
the script's: the three barriers against W, then train x train near duplicates in sorted id
order (the first of a pair stays).
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Mapping

import pytest

from artigo_env.corpus.decontam import (
    NearDuplicateIndex,
    TrainDoc,
    WDoc,
    decontaminate,
    gram_tokens,
    shingle_hashes,
)

VOCAB = (  # noqa: SIM905 (same layout as the reference script)
    "casa rio pedra vento livro estrada cidade campo noite manhã sol lua mar "
    "porto ilha serra vale ponte torre igreja praça mercado escola música "
    "pintura história ciência máquina ferro madeira árvore flor fruto animal "
    "pássaro peixe cavalo homem mulher criança povo reino guerra paz lei governo "
    "trabalho viagem caminho sinal número forma corpo água fogo terra ar luz "
    "sombra cor som palavra nome coração memória tempo ano dia século norte sul "
    "leste oeste grande pequeno antigo novo alto baixo forte rápido lento claro "
    "escuro quente frio verde azul branco negro construiu descobriu levou trouxe "
    "viu disse fez deu ficou passou chegou saiu voltou abriu fechou escreveu "
    "cantou plantou colheu vendeu"
).split()
STOP = "o a os as de do da dos das em no na um uma e que com por para se".split()  # noqa: SIM905
COMMON_PHRASE = (
    "segundo o instituto brasileiro de geografia e estatística a população estimada do "
    "município era de"
)

CONTAMINATED = "contaminated"
TOLERATED = "tolerated"
ONLY_FOLDED = "only_folded"

Texts = dict[str, tuple[str, str]]


def random_text(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(STOP) if rng.random() < 0.3 else rng.choice(VOCAB) for _ in range(n))


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def build_cases(seed: int = 42) -> tuple[Texts, Texts, dict[str, str | None]]:
    """Training texts, W texts and the expected outcome per training id (``None`` = kept)."""
    rng = random.Random(seed)
    w = {f"av{k}": (f"Verbete {k}", random_text(rng, 600)) for k in range(5)}
    w["av5"] = ("al-Khwarizmi", random_text(rng, 600))
    title, body = w["av0"]
    w["av0"] = (title, body + " " + COMMON_PHRASE + " " + random_text(rng, 50))

    train: Texts = {}
    expected: dict[str, str | None] = {}

    for k in range(30):  # (1) clean
        train[f"limpo{k:02d}"] = (f"Documento {k}", random_text(rng, 400))
        expected[f"limpo{k:02d}"] = None

    train["copia_integral"] = ("Outro título", w["av1"][1])  # (2)
    expected["copia_integral"] = CONTAMINATED

    words = w["av2"][1].split()  # (3)
    edited = [rng.choice(VOCAB) if rng.random() < 0.05 else word for word in words]
    train["copia_5pct"] = ("Título editado", " ".join(edited))
    expected["copia_5pct"] = CONTAMINATED

    excerpt13 = " ".join(w["av3"][1].split()[100:113])  # (4)
    train["trecho_13"] = (
        "Com trecho",
        random_text(rng, 200) + " " + excerpt13 + " " + random_text(rng, 200),
    )
    expected["trecho_13"] = CONTAMINATED

    excerpt12 = " ".join(w["av3"][1].split()[300:312])  # (5)
    train["trecho_12"] = (
        "Trecho curto",
        random_text(rng, 200) + " " + excerpt12 + " " + random_text(rng, 200),
    )
    # Below 13 words it is normally kept; if a neighbouring word coincides by chance it becomes
    # a true 13-gram and removal is acceptable.
    expected["trecho_12"] = TOLERATED

    for k in range(12):  # (6) frequent phrase, also present in W
        train[f"comum{k:02d}"] = (
            f"Município {k}",
            random_text(rng, 200) + " " + COMMON_PHRASE + " " + random_text(rng, 200),
        )
        expected[f"comum{k:02d}"] = None

    train["titulo_caixa"] = ("Al-Khwarizmi", random_text(rng, 400))  # (7)
    expected["titulo_caixa"] = CONTAMINATED

    # (8) full copy without accents: a long text keeps 13-grams with no accented word.
    train["sem_acentos"] = ("Cópia desacentuada", strip_accents(w["av4"][1]))
    expected["sem_acentos"] = CONTAMINATED
    # (9) a 13-word excerpt with an accented word, accents stripped: only the folded index.
    words4 = w["av4"][1].split()
    start = next(
        i
        for i in range(len(words4) - 13)
        if any(strip_accents(word) != word for word in words4[i : i + 13])
    )
    excerpt_folded = strip_accents(" ".join(words4[start : start + 13]))
    train["trecho_sem_acentos"] = (
        "Trecho desacentuado",
        random_text(rng, 200) + " " + excerpt_folded + " " + random_text(rng, 200),
    )
    expected["trecho_sem_acentos"] = ONLY_FOLDED

    base = random_text(rng, 500)  # (10) train x train near duplicate (1% of words changed)
    base_words = base.split()
    changed = set(rng.sample(range(len(base_words)), 5))
    near = [
        rng.choice([v for v in VOCAB if v != word]) if k in changed else word
        for k, word in enumerate(base_words)
    ]
    train["dup_a"] = ("Duplicata A", base)
    train["dup_b"] = ("Duplicata B", " ".join(near))
    expected["dup_a"] = None
    expected["dup_b"] = CONTAMINATED

    # (11) systematic paraphrase (one word in ten): no 13-gram survives; a declared limit.
    words1 = w["av1"][1].split()
    paraphrase = [
        rng.choice([v for v in VOCAB if v != word]) if i % 10 == 5 else word
        for i, word in enumerate(words1)
    ]
    train["parafrase_10"] = ("Paráfrase", " ".join(paraphrase))
    expected["parafrase_10"] = None
    return train, w, expected


def run_flow(train: Texts, w: Texts, *, fold_accents: bool) -> dict[str, str]:
    """Removed training ids and their reason (barriers against W, then train x train)."""
    ids = list(train)
    docs = [TrainDoc(i, train[i][0], gram_tokens(train[i][1])) for i in ids]
    w_docs = [WDoc(title, gram_tokens(body)) for title, body in w.values()]
    verdicts = decontaminate(docs, w_docs, fold_accents=fold_accents)
    removed = {v.doc_id: v.reason for v in verdicts if v.reason is not None}

    dedup = NearDuplicateIndex()
    position = {doc_id: k for k, doc_id in enumerate(sorted(ids))}
    tokens = {doc.doc_id: doc.tokens for doc in docs}
    for doc_id in sorted(ids):
        if doc_id in removed:
            continue
        shingles = shingle_hashes(tokens[doc_id])
        match = dedup.match(shingles)
        if match is not None:
            removed[doc_id] = f"near_duplicate_of:{sorted(ids)[match]}"
        else:
            dedup.add(position[doc_id], shingles)
    return {doc_id: str(reason) for doc_id, reason in removed.items()}


def failures(expected: Mapping[str, str | None], removed: Mapping[str, str], fold: bool) -> list:
    out = []
    for doc_id in sorted(expected):
        outcome = expected[doc_id]
        gone = doc_id in removed
        if outcome == CONTAMINATED:
            ok = gone
        elif outcome is None:
            ok = not gone
        elif outcome == ONLY_FOLDED:
            ok = gone == fold
        else:
            ok = True
        if not ok:
            out.append((fold, doc_id, outcome, removed.get(doc_id, "kept")))
    return out


def failures_for_seed(seed: int) -> list:
    train, w, expected = build_cases(seed)
    found = []
    for fold in (False, True):
        found += failures(expected, run_flow(train, w, fold_accents=fold), fold)
    return found


@pytest.fixture(scope="module")
def outcome() -> dict[bool, dict[str, str]]:
    train, w, _ = build_cases(42)
    return {fold: run_flow(train, w, fold_accents=fold) for fold in (False, True)}


BOTH = pytest.mark.parametrize("fold", [False, True], ids=["plain", "folded"])


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case01_clean_documents_kept(outcome, fold):
    removed = outcome[fold]
    assert [f"limpo{k:02d}" for k in range(30) if f"limpo{k:02d}" in removed] == []


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case02_full_copy_with_other_title_removed(outcome, fold):
    assert outcome[fold]["copia_integral"] == "w_13gram"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008", "REQ-COR-009")
@BOTH
def test_case03_copy_with_5pct_edited_removed(outcome, fold):
    assert outcome[fold]["copia_5pct"] in {"w_13gram", "w_near_duplicate"}


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case04_13_word_excerpt_removed(outcome, fold):
    assert outcome[fold]["trecho_13"] == "w_13gram"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case05_12_word_excerpt_kept(outcome, fold):
    assert "trecho_12" not in outcome[fold]


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case06_frequent_phrase_shared_with_w_kept(outcome, fold):
    removed = outcome[fold]
    assert [f"comum{k:02d}" for k in range(12) if f"comum{k:02d}" in removed] == []


@pytest.mark.fast
@pytest.mark.req("REQ-COR-007")
@BOTH
def test_case07_title_first_letter_case_removed(outcome, fold):
    assert outcome[fold]["titulo_caixa"] == "w_title"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case08_full_copy_without_accents_removed_by_default_index(outcome, fold):
    assert outcome[fold]["sem_acentos"] == "w_13gram"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
def test_case09_excerpt_without_accents_needs_folding(outcome):
    assert "trecho_sem_acentos" not in outcome[False]
    assert outcome[True]["trecho_sem_acentos"] == "w_13gram"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-009", "REQ-COR-011")
@BOTH
def test_case10_train_near_duplicate_removed_first_kept(outcome, fold):
    removed = outcome[fold]
    assert "dup_a" not in removed
    assert removed["dup_b"] == "near_duplicate_of:dup_a"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@BOTH
def test_case11_systematic_paraphrase_escapes(outcome, fold):
    # Declared limit (SDD 06 §7.5): one word in ten replaced leaves no intact 13-gram.
    assert "parafrase_10" not in outcome[fold]


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008")
@pytest.mark.parametrize("seed", range(5))
def test_sweep_5_seeds(seed):
    assert failures_for_seed(seed) == []


@pytest.mark.slow
@pytest.mark.req("REQ-COR-008")
def test_sweep_50_seeds():
    bad = {seed: found for seed in range(50) if (found := failures_for_seed(seed))}
    assert bad == {}
