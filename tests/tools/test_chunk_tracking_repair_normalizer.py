"""The repair tool replays extraction with the operator's entity-name normalizer.

It parses cached LLM output, which keys entity names through the selected
normalizer. Run without the server's selection it would rebuild names the
normalizer would have merged, so it reads the same environment variable.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lightrag import entity_name_normalizer as normalizer
from lightrag.tools import chunk_tracking_repair

pytestmark = pytest.mark.offline


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.setattr(normalizer, "_REGISTRY", {})
    monkeypatch.setattr(normalizer, "_DUPLICATES", set())
    monkeypatch.setattr(normalizer, "_loaded", False)
    monkeypatch.setattr(normalizer, "_failures", [])
    monkeypatch.setattr(normalizer, "_active", None)
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: [])
    monkeypatch.delenv("CUSTOM_ENTITY_NAME_NORMALIZER", raising=False)
    monkeypatch.setattr(chunk_tracking_repair, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(chunk_tracking_repair, "setup_logger", lambda *a, **k: None)
    monkeypatch.setattr(chunk_tracking_repair, "_parse_args", lambda: SimpleNamespace())

    seen = []

    async def run(args):
        seen.append(normalizer.get_active_entity_name_normalizer())
        return True

    monkeypatch.setattr(chunk_tracking_repair, "run", run)
    return seen


def register(monkeypatch, name="acme"):
    monkeypatch.setitem(
        sys.modules, "repair_normalizer_impl", SimpleNamespace(normalize=str.lower)
    )

    def provider():
        normalizer.register_entity_name_normalizer(
            normalizer.EntityNameNormalizerSpec(
                name, "repair_normalizer_impl:normalize", "1", "Acme"
            )
        )

    entry = SimpleNamespace(
        name=name,
        value="provider:register",
        dist=SimpleNamespace(name="dist-acme", version="1"),
        load=lambda: provider,
    )
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: [entry])


def test_the_selected_normalizer_is_active_when_the_tool_runs(tool, monkeypatch):
    register(monkeypatch)
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "acme")

    chunk_tracking_repair.main()

    assert [n.spec.name for n in tool] == ["acme"]


def test_without_a_selection_no_normalizer_is_active(tool):
    chunk_tracking_repair.main()

    assert tool == [None]


def test_an_invalid_selection_stops_the_tool_before_it_runs(tool, monkeypatch):
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "missing")

    with pytest.raises(SystemExit) as exit_info:
        chunk_tracking_repair.main()

    assert exit_info.value.code == 1
    assert tool == []
