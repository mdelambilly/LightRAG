"""Exercise the shared uvicorn/Gunicorn app factory, not just a helper."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lightrag import entity_name_normalizer as normalizer
from lightrag.api import config

pytestmark = pytest.mark.offline


@pytest.fixture
def startup(monkeypatch, tmp_path, lightrag_caplog):
    monkeypatch.setattr(config, "_global_args", None)
    monkeypatch.setattr(config, "_initialized", False)
    monkeypatch.setattr(normalizer, "_REGISTRY", {})
    monkeypatch.setattr(normalizer, "_DUPLICATES", set())
    monkeypatch.setattr(normalizer, "_loaded", False)
    monkeypatch.setattr(normalizer, "_failures", [])
    monkeypatch.setattr(normalizer, "_active", None)
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: [])
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "")
    monkeypatch.setenv("CUSTOM_CHUNKER", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("UI_TEMPLATES_DIR", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--working-dir",
            str(tmp_path / "rag"),
            "--input-dir",
            str(tmp_path / "inputs"),
        ],
    )
    args = config.initialize_config(force=True)
    from lightrag.api import lightrag_server as server

    captured = []

    class Constructed(Exception):
        pass

    def construct(**kwargs):
        captured.append(kwargs)
        raise Constructed

    monkeypatch.setattr(server, "LightRAG", construct)
    monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
    monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)
    return server, args, captured, Constructed


def install_provider(monkeypatch, implementation, name="acme"):
    monkeypatch.setitem(
        sys.modules,
        "startup_normalizer_impl",
        SimpleNamespace(normalize=implementation),
    )

    def register():
        normalizer.register_entity_name_normalizer(
            normalizer.EntityNameNormalizerSpec(
                name, "startup_normalizer_impl:normalize", "1", "Acme"
            )
        )

    return SimpleNamespace(
        name=name,
        value="provider:register",
        dist=SimpleNamespace(name="dist-acme", version="1"),
        load=lambda: register,
    )


@pytest.mark.parametrize("factory", ["create_app", "get_application"])
def test_discovered_normalizer_is_active_before_the_constructor(
    startup, monkeypatch, factory
):
    server, args, captured, constructed = startup
    entry = install_provider(monkeypatch, str.lower)
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: [entry])
    args.custom_entity_name_normalizer = "acme"

    with pytest.raises(constructed):
        getattr(server, factory)(args)

    active = normalizer.get_active_entity_name_normalizer()
    assert active is not None and active.spec.name == "acme"
    assert normalizer.apply_entity_name_normalizer("ACME") == "acme"


def test_unset_leaves_no_normalizer_active(startup):
    server, args, captured, constructed = startup

    with pytest.raises(constructed):
        server.create_app(args)

    assert normalizer.get_active_entity_name_normalizer() is None
    assert normalizer.apply_entity_name_normalizer("ACME") == "ACME"


def test_unset_turns_off_a_normalizer_left_by_an_earlier_app(startup, monkeypatch):
    server, args, captured, constructed = startup
    entry = install_provider(monkeypatch, str.lower)
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: [entry])
    args.custom_entity_name_normalizer = "acme"
    with pytest.raises(constructed):
        server.create_app(args)
    assert normalizer.get_active_entity_name_normalizer() is not None

    args.custom_entity_name_normalizer = ""
    with pytest.raises(constructed):
        server.create_app(args)

    assert normalizer.get_active_entity_name_normalizer() is None


@pytest.mark.parametrize(
    "name,match",
    [("missing", "unknown normalizer"), ("pkg:func", "import paths")],
)
def test_invalid_selection_fails_before_constructor(startup, name, match):
    server, args, captured, _ = startup
    args.custom_entity_name_normalizer = name

    with pytest.raises(ValueError, match=match):
        server.create_app(args)

    assert captured == []
    assert normalizer.get_active_entity_name_normalizer() is None


def test_invalid_selection_prints_a_boxed_notice_and_still_aborts(startup, monkeypatch):
    """The boxed output is presentation only: the ValueError must still leave
    ``create_app``, so an unselectable normalizer cannot reach a running server.
    ASCIIColors writes through its own stream rather than the one capsys
    replaces, so collect the arguments it is handed instead.
    """
    server, args, captured, _ = startup
    colors = MagicMock()
    monkeypatch.setattr(server, "ASCIIColors", colors)
    args.custom_entity_name_normalizer = "pkg:func"

    with pytest.raises(ValueError, match="import paths"):
        server.create_app(args)

    notice = "\n".join(str(a) for call in colors.mock_calls for a in call.args)
    assert "ERROR: invalid CUSTOM_ENTITY_NAME_NORMALIZER selection" in notice
    assert "import paths are not supported" in notice


@pytest.mark.parametrize(
    "selected,outcome",
    [
        ("acme", "selected normalizer 'acme' validated"),
        (
            "partial",
            "CUSTOM_ENTITY_NAME_NORMALIZER='partial' could not be activated; "
            "startup aborted",
        ),
        (
            "",
            "CUSTOM_ENTITY_NAME_NORMALIZER is unset; entity names are unchanged",
        ),
    ],
)
def test_failed_provider_log_alone_describes_the_selection_outcome(
    startup, monkeypatch, lightrag_caplog, selected, outcome
):
    server, args, captured, constructed = startup
    good = install_provider(monkeypatch, str.lower)

    def broken():
        normalizer.register_entity_name_normalizer(
            normalizer.EntityNameNormalizerSpec("partial", "absent:n", "1", "Partial")
        )
        raise RuntimeError("provider failed")

    entries = [
        good,
        SimpleNamespace(
            name="broken",
            value="broken:register",
            dist=SimpleNamespace(name="dist-broken", version="1"),
            load=lambda: broken,
        ),
    ]
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: entries)
    args.custom_entity_name_normalizer = selected

    with pytest.raises(ValueError if selected == "partial" else constructed):
        server.create_app(args)

    errors = [
        r.getMessage()
        for r in lightrag_caplog.records
        if "skipped failed plugin" in r.getMessage()
    ]
    assert len(errors) == 1
    assert "broken:register" in errors[0] and "provider failed" in errors[0]
    assert outcome in errors[0]
