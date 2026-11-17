"""ArtigoEnv ``services`` layer (see docs/sdd/02, section B.2).

Import the builder from ``artigo_env.services.knowledge_build`` (it reads the corpus); the reader
below is what agents use.
"""

from artigo_env.services.knowledge import (
    KnowledgeMismatchError,
    KnowledgeStore,
    find_knowledge,
)

__all__ = ["KnowledgeMismatchError", "KnowledgeStore", "find_knowledge"]
