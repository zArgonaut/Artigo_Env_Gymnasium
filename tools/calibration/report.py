"""Per-session report (SDD 03 §7.4). Internal document, written in Portuguese.

Never prints the article title or text: only hashes and counts.
"""

from __future__ import annotations

from typing import Any

from artigo_env.text.tokenizer import is_header_line, tokenize
from tools.calibration.align import Alignment
from tools.calibration.session import RecordedSession, extract_article


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _optional(value: object) -> str:
    return "—" if value is None else _cell(value)


def build_report(session: RecordedSession, alignment: Alignment, fixture: dict[str, Any]) -> str:
    """Deterministic Markdown: no dates, no absolute paths."""
    meta = session.meta
    lines = [
        f"# Relatório da sessão {fixture['source_session']}",
        "",
        f"Versão de calibração: {fixture['calibration_version']}",
        "",
        "## Sessão",
        "",
        f"- Modo: {_optional(meta.get('mode'))}",
        f"- Hash do bundle: {_optional(meta.get('bundle_hash'))}",
        f"- Completa: {'sim' if meta.get('complete') else 'não'}",
        f"- Número de palpites: {len(fixture['guesses'])}",
        "",
        "## Artigo",
        "",
        f"- SHA-256 do texto: {fixture['article_ref']['sha256']}",
        f"- SHA-256 do título: {fixture['article_ref']['title_sha256']}",
    ]
    article = extract_article(session)
    if article is not None:
        title, text = article
        headers = sum(1 for line in text.split("\n") if is_header_line(line))
        lines += [
            f"- Tokens no título: {len(tokenize(title))}",
            f"- Tokens no corpo: {len(tokenize(text))}",
            f"- Cabeçalhos de seção: {headers}",
            f"- Parênteses no título: {'sim' if '(' in title or ')' in title else 'não'}",
        ]
    lines += ["", "## Divergências", ""]
    if alignment.divergences:
        lines += ["| Tipo | Elemento DOM | Token | Detalhe |", "|---|---|---|---|"]
        lines += [
            f"| {item.kind} | {_optional(item.dom_index)} | {_optional(item.token_index)} "
            f"| {_cell(item.detail)} |"
            for item in alignment.divergences
        ]
    else:
        lines.append("Nenhuma")
    lines += [
        "",
        "## Palpites",
        "",
        "| Índice | Palpite | Contou | Quantidade | Mensagem | Não confiável |",
        "|---|---|---|---|---|---|",
    ]
    for index, guess in enumerate(fixture["guesses"]):
        expected = guess["expected"]
        lines.append(
            f"| {index} | {_cell(guess['raw'])} | {'sim' if expected['counted'] else 'não'} "
            f"| {expected['count']} | {_optional(expected['message'])} "
            f"| {'sim' if expected.get('unreliable') else 'não'} |"
        )
    return "\n".join(lines) + "\n"
