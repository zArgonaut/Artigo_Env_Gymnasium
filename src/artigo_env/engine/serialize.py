"""JSON-compatible state serialization (REQ-ENG-011)."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from artigo_env.engine.types import GameState, GuessRecord

FORMAT_VERSION = 1


def serialize_state(state: GameState) -> dict[str, Any]:
    """Return a dict that survives ``json.dumps``/``json.loads`` unchanged."""
    packed = np.packbits(state.revealed)
    return {
        "format_version": FORMAT_VERSION,
        "size": int(len(state.revealed)),
        "revealed": base64.b64encode(packed.tobytes()).decode("ascii"),
        "partial": [[int(pos), int(mask)] for pos, mask in sorted(state.partial.items())],
        "guessed": sorted(state.guessed),
        "history": [
            {"key": r.key, "form": r.form, "kind": r.kind, "counted": r.counted, "count": r.count}
            for r in state.history
        ],
        "n_counted": state.n_counted,
        "status": state.status,
    }


def deserialize_state(data: dict[str, Any]) -> GameState:
    """Rebuild a ``GameState``; raises ``ValueError`` on malformed input."""
    try:
        if data["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported state format_version: {data['format_version']!r}")
        size = int(data["size"])
        raw = np.frombuffer(base64.b64decode(data["revealed"]), dtype=np.uint8)
        if size < 0 or len(raw) != (size + 7) // 8:
            raise ValueError("revealed bits do not match the declared size")
        revealed = np.unpackbits(raw, count=size).astype(np.bool_)
        status = data["status"]
        if status not in ("playing", "won"):
            raise ValueError(f"invalid status: {status!r}")
        return GameState(
            revealed=revealed,
            partial={int(pos): int(mask) for pos, mask in data["partial"]},
            guessed=frozenset(str(key) for key in data["guessed"]),
            history=tuple(
                GuessRecord(
                    str(r["key"]),
                    str(r["form"]),
                    str(r["kind"]),
                    bool(r["counted"]),
                    int(r["count"]),
                )
                for r in data["history"]
            ),
            n_counted=int(data["n_counted"]),
            status=status,
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"malformed game state: {error!r}") from error
