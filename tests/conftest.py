"""Shared pytest configuration.

Investigations write to operational memory. Tests must never write to the real
``data/operational_memory.db``, so the database path is redirected to a session
temporary file for the whole run. Individual tests can still pass an explicit
``memory_db`` path when they need a completely fresh database.
"""

import os
import tempfile

_SESSION_DB = os.path.join(tempfile.mkdtemp(prefix="opmem-tests-"), "test_memory.db")

os.environ["OPERATIONAL_MEMORY_DB"] = _SESSION_DB

# Marks this process as a test session: entrypoints check this before loading a
# developer's real .env file, and every provider variable is cleared below.
os.environ["OI_TEST_SESSION"] = "1"

# Reasoning-layer credentials must not leak from the developer's shell into
# the tests; every provider variable is cleared for the whole session.
for _name in (
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_PROVIDER",
    "LLM_FALLBACK_PROVIDER",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "TYPESAFE_API_KEY",
    "TYPESAFE_MODEL",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_SUPPORT_THRESHOLD",
    "TYPESAFE_REJECT_THRESHOLD",
):
    os.environ.pop(_name, None)