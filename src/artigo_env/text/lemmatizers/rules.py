"""Interpreter for suffix-rewriting rules (SDD 05 §5, DEC-040).

Three constructs, mirroring the reference game's lemmatizer:

* ``SuffixRuleSet``: one pattern ``^(prefix)(t1|t2|...)(suffix)$``; the first matching target
  decides, a whole-word exception blocks it, otherwise the output is ``prefix + replacement``.
* ``IndexedRuleSet``: rules grouped by the first character of the target, one
  ``SuffixRuleSet`` per group.
* ``FirstMatch``: the first member that applies wins.

Targets, prefixes, suffixes and exceptions are regex fragments (the ``regex`` package). Patterns
are compiled lazily, once per rule set, so tables with thousands of alternatives load fast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import regex

Rule = tuple[str, str, str]
"""``(target, replacement, exceptions)``; ``exceptions`` may be empty."""

_RULE_GROUP = "_r"


class RuleNode(Protocol):
    """A rule set: ``apply`` returns the rewritten word or ``None`` when it does not apply."""

    def apply(self, word: str) -> str | None: ...

    def to_dict(self) -> dict[str, object]: ...


class SuffixRuleSet:
    """``Ft`` of the reference: one alternation of targets between a prefix and a suffix."""

    def __init__(self, prefix: str, suffix: str, rules: Sequence[Rule]) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.rules: tuple[Rule, ...] = tuple((t, r, e) for t, r, e in rules)
        self._pattern: regex.Pattern[str] | None = None
        self._indexes: list[int] | None = None
        # The reference keeps replacements and exceptions in maps keyed by target, so when a
        # target is listed twice the LAST replacement wins and the last NON-EMPTY exception
        # list stays in force.
        self._replacement: dict[str, str] = {}
        self._exception_source: dict[str, str] = {}
        for target, replacement, exceptions in self.rules:
            self._replacement[target] = replacement
            if exceptions:
                self._exception_source[target] = exceptions
        self._exceptions: dict[str, regex.Pattern[str]] = {}

    def _compile(self) -> regex.Pattern[str]:
        if self._pattern is None:
            alternatives = "|".join(
                f"(?P<{_RULE_GROUP}{i}>{target})" for i, (target, _, _) in enumerate(self.rules)
            )
            self._pattern = regex.compile(
                f"^(?P<pre>{self.prefix})(?:{alternatives})(?:{self.suffix})$"
            )
            groups = self._pattern.groupindex
            self._indexes = [groups[f"{_RULE_GROUP}{i}"] for i in range(len(self.rules))]
        return self._pattern

    def _rule_index(self, match: regex.Match[str]) -> int:
        # The suffix is not a capturing group, so normally the rule's own group closes last and
        # ``lastgroup`` names it; otherwise (groups inside the suffix) scan the rule groups.
        name = match.lastgroup
        if name is not None and name.startswith(_RULE_GROUP):
            return int(name[len(_RULE_GROUP) :])
        assert self._indexes is not None
        for i, group in enumerate(self._indexes):
            if match.start(group) != -1:
                return i
        raise AssertionError("a match without a participating rule group")

    def apply(self, word: str) -> str | None:
        if not self.rules:
            return None
        match = self._compile().match(word)
        if match is None:
            return None
        target = self.rules[self._rule_index(match)][0]
        exceptions = self._exception_source.get(target)
        if exceptions:
            blocker = self._exceptions.get(target)
            if blocker is None:
                blocker = self._exceptions[target] = regex.compile(f"^({exceptions})$")
            if blocker.match(word) is not None:
                return None
        return match.group("pre") + self._replacement[target]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "suffix",
            "prefix": self.prefix,
            "suffix": self.suffix,
            "rules": [list(rule) for rule in self.rules],
        }


class IndexedRuleSet:
    """``At`` of the reference: rules grouped by the first character of the target."""

    def __init__(self, prefix: str, suffix: str, rules: Sequence[Rule]) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.rules: tuple[Rule, ...] = tuple((t, r, e) for t, r, e in rules)
        grouped: dict[str, list[Rule]] = {}
        for rule in self.rules:
            grouped.setdefault(rule[0][:1], []).append(rule)
        self._groups: dict[str, SuffixRuleSet] = {
            key: SuffixRuleSet(prefix, suffix, group) for key, group in grouped.items()
        }
        self._prefix_pattern: regex.Pattern[str] | None = None

    def apply(self, word: str) -> str | None:
        if self._prefix_pattern is None:
            self._prefix_pattern = regex.compile(f"^(?:{self.prefix})")
        match = self._prefix_pattern.match(word)
        consumed = len(match.group()) if match is not None else 0
        rest = word[consumed:] if consumed else word
        t, n = rest[:1], word[:1]
        group = self._groups.get(t)
        if group is not None:
            result = group.apply(word)
            if result is not None:
                return result
        if n != t:
            group = self._groups.get(n)
            if group is not None:
                return group.apply(word)
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "indexed",
            "prefix": self.prefix,
            "suffix": self.suffix,
            "rules": [list(rule) for rule in self.rules],
        }


class FirstMatch:
    """``Ot`` of the reference: the first member that applies wins."""

    def __init__(self, members: Sequence[RuleNode]) -> None:
        self.members: tuple[RuleNode, ...] = tuple(members)

    def apply(self, word: str) -> str | None:
        for member in self.members:
            result = member.apply(word)
            if result is not None:
                return result
        return None

    def to_dict(self) -> dict[str, object]:
        return {"type": "first", "members": [member.to_dict() for member in self.members]}


def _text(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"rule node field {key!r} must be a string")
    return value


def _rules(data: Mapping[str, object]) -> tuple[Rule, ...]:
    raw = data.get("rules")
    if not isinstance(raw, list | tuple):
        raise ValueError("rule node field 'rules' must be a list")
    out: list[Rule] = []
    for item in raw:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 3
            or not all(isinstance(part, str) for part in item)
        ):
            raise ValueError("each rule must be [target, replacement, exceptions] of strings")
        out.append((str(item[0]), str(item[1]), str(item[2])))
    return tuple(out)


def node_from_dict(data: Mapping[str, object]) -> RuleNode:
    """Build a rule node from its ``to_dict`` form."""
    kind = data.get("type")
    if kind == "suffix":
        return SuffixRuleSet(_text(data, "prefix"), _text(data, "suffix"), _rules(data))
    if kind == "indexed":
        return IndexedRuleSet(_text(data, "prefix"), _text(data, "suffix"), _rules(data))
    if kind == "first":
        members = data.get("members")
        if not isinstance(members, list | tuple):
            raise ValueError("rule node field 'members' must be a list")
        built: list[RuleNode] = []
        for member in members:
            if not isinstance(member, Mapping):
                raise ValueError("each member must be an object")
            built.append(node_from_dict(member))
        return FirstMatch(built)
    raise ValueError(f"unknown rule node type: {kind!r}")
