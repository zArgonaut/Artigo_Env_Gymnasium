import pytest

from artigo_env.text.lemmatizers.base import RuleLemmatizer
from artigo_env.text.lemmatizers.rules import (
    FirstMatch,
    IndexedRuleSet,
    SuffixRuleSet,
    node_from_dict,
)
from artigo_env.text.lemmatizers.tables import LemmaTables

pytestmark = pytest.mark.fast

NEVER = SuffixRuleSet("", "", ())


def tables(**stages) -> LemmaTables:
    nodes = {"irregular": NEVER, "regular": NEVER, "plural": NEVER, "gender": NEVER} | stages
    return LemmaTables("t", "synthetic", **nodes)


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_lazy_prefix_makes_longest_target_win():
    rules = SuffixRuleSet(".*?", "", (("s", "", ""), ("es", "", ""), ("ões", "ão", "")))
    assert rules.apply("leões") == "leão"
    assert rules.apply("lugares") == "lugar"
    assert rules.apply("casas") == "casa"
    assert rules.apply("casa") is None


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_first_alternative_in_order_is_reported_when_two_match_same_text():
    # Two DIFFERENT targets that can match the same text: the first in rule order is used.
    # (Identical targets follow the reference's map semantics, tested further below.)
    rules = SuffixRuleSet(".*?", "", (("s", "A", ""), ("[sz]", "B", "")))
    assert rules.apply("cas") == "caA"
    assert rules.apply("caz") == "caB"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_whole_word_exception_blocks_the_rule():
    rules = SuffixRuleSet(".*?", "", (("s", "", "lápis|mais"),))
    assert rules.apply("lápis") is None
    assert rules.apply("mais") is None
    assert rules.apply("maiss") == "mais"  # exception needs the whole word
    assert rules.apply("casas") == "casa"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_matched_suffix_is_dropped_from_output():
    rules = SuffixRuleSet("(torn)", "(?:ou|aram)", (("", "ar", ""),))
    assert rules.apply("tornou") == "tornar"
    assert rules.apply("tornaram") == "tornar"
    assert rules.apply("torna") is None


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_suffix_with_inner_groups_still_finds_the_rule():
    rules = SuffixRuleSet(".*?", "(a|b)(c|d)", (("x", "1", ""), ("y", "2", "")))
    assert rules.apply("zybd") == "z2"
    assert rules.apply("zxac") == "z1"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_indexed_tries_letter_after_prefix_then_first_letter():
    rules = IndexedRuleSet(
        "(?:re|des)?", "ar", (("fa", "fazer", ""), ("re", "R", ""), ("d", "D", ""))
    )
    # prefix "re" is consumed, the rest starts with "f" -> group "f" applies
    assert rules.apply("refaar") == "refazer"
    # prefix "des" is consumed, the rest starts with "d"; group "d" applies directly
    assert rules.apply("desdar") == "desD"
    # the rest starts with "z", group "z" does not exist and neither does one for "d"/"r"
    assert rules.apply("zzar") is None


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_indexed_falls_back_to_group_of_first_letter_of_word():
    # "reg": the prefix "re" is consumed and the rest starts with "g", which has no group;
    # the word's own first letter "r" does.
    rules = IndexedRuleSet("(?:re)?", "", (("reg", "REG", ""), ("x", "X", "")))
    assert rules.apply("reg") == "REG"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_indexed_prefix_that_matches_nothing_uses_whole_word():
    rules = IndexedRuleSet("(?:re)?", "", (("x", "y", ""),))
    assert rules.apply("x") == "y"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_first_match_stops_at_first_result():
    first = SuffixRuleSet("", "", (("a", "1", ""),))
    second = SuffixRuleSet("", "", (("a", "2", ""), ("b", "3", "")))
    chain = FirstMatch([first, second])
    assert chain.apply("a") == "1"
    assert chain.apply("b") == "3"
    assert chain.apply("c") is None


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_verb_rule_blocks_nominal_rules():
    verb = SuffixRuleSet(".*?", "", (("ou", "ar", ""),))
    plural = SuffixRuleSet(".*?", "", (("s", "", ""),))
    lemmatizer = RuleLemmatizer("t", tables(regular=verb, plural=plural))
    assert lemmatizer.lemma("tornou") == "tornar"
    assert lemmatizer.lemma("TORNOU") == "tornar"
    assert lemmatizer.lemma("casas") == "casa"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_irregular_wins_over_regular():
    irregular = SuffixRuleSet("", "", (("tem", "ter", ""),))
    regular = SuffixRuleSet(".*?", "", (("m", "X", ""),))
    lemmatizer = RuleLemmatizer("t", tables(irregular=irregular, regular=regular))
    assert lemmatizer.lemma("tem") == "ter"
    assert lemmatizer.lemma("vem") == "veX"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_plural_and_gender_chain():
    plural = SuffixRuleSet(".*?", "", (("s", "", ""),))
    gender = SuffixRuleSet(".*?", "", (("na", "no", ""),))
    lemmatizer = RuleLemmatizer("t", tables(plural=plural, gender=gender))
    assert lemmatizer.lemma("pequenas") == "pequeno"
    assert lemmatizer.lemma("pequena") == "pequeno"
    assert lemmatizer.lemma("casa") == "casa"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_dict_roundtrip():
    node = FirstMatch(
        [
            SuffixRuleSet("(a)", "b", (("x", "y", "z"),)),
            IndexedRuleSet(".*?", "", (("ab", "c", ""), ("d", "e", "f"))),
        ]
    )
    data = node.to_dict()
    assert node_from_dict(data).to_dict() == data
    assert data["type"] == "first"


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_unknown_node_type_and_bad_rules():
    with pytest.raises(ValueError):
        node_from_dict({"type": "nope"})
    with pytest.raises(ValueError):
        node_from_dict({"type": "suffix", "prefix": "", "suffix": "", "rules": [["a", "b"]]})


@pytest.mark.req("REQ-TXT-016")
def test_changing_a_rule_changes_content_hash():
    a = RuleLemmatizer("t", tables(plural=SuffixRuleSet(".*?", "", (("s", "", ""),))))
    b = RuleLemmatizer("t", tables(plural=SuffixRuleSet(".*?", "", (("s", "x", ""),))))
    same = RuleLemmatizer("t", tables(plural=SuffixRuleSet(".*?", "", (("s", "", ""),))))
    assert a.content_hash() != b.content_hash()
    assert a.content_hash() == same.content_hash()


@pytest.mark.req("REQ-TXT-014")
def test_cache_does_not_change_results():
    plural = SuffixRuleSet(".*?", "", (("s", "", ""),))
    cached = RuleLemmatizer("t", tables(plural=plural), cache_size=8)
    uncached = RuleLemmatizer("t", tables(plural=plural), cache_size=0)
    words = ["casas", "casa", "casas", "livros", "x"] * 3
    assert [cached.lemma(w) for w in words] == [uncached.lemma(w) for w in words]
    assert cached.lemma.cache_info().hits > 0


@pytest.mark.req("DEC-040")
def test_repeated_target_last_replacement_wins_and_exceptions_persist():
    # The reference stores rules in maps keyed by target: a later duplicate overrides the
    # replacement, while an earlier non-empty exception list is kept.
    rules = SuffixRuleSet(
        ".*?", "", (("vir", "ver", "elvir"), ("vimos", "ver", ""), ("vir", "vir", ""))
    )
    assert rules.apply("vir") == "vir"
    assert rules.apply("revir") == "revir"
    assert rules.apply("vimos") == "ver"
    assert rules.apply("elvir") is None
