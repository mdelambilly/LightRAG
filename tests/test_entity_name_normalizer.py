"""Closed-set selection, lazy imports, isolated discovery and the call contract
of third-party entity-name normalizers."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag import entity_name_normalizer as normalizer
from lightrag.constants import (
    DEFAULT_ENTITY_NAME_MAX_BYTES,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)

pytestmark = pytest.mark.offline


def test_registry_import_does_not_load_a_plugin_implementation():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lightrag.entity_name_normalizer import "
            "EntityNameNormalizerSpec, register_entity_name_normalizer; "
            "register_entity_name_normalizer("
            "EntityNameNormalizerSpec('demo', 'absent:normalize', '1', 'Demo')); "
            "assert 'absent' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, lightrag_caplog):
    monkeypatch.setattr(normalizer, "_REGISTRY", {})
    monkeypatch.setattr(normalizer, "_DUPLICATES", set())
    monkeypatch.setattr(normalizer, "_loaded", False)
    monkeypatch.setattr(normalizer, "_failures", [])
    monkeypatch.setattr(normalizer, "entry_points", lambda **kwargs: [])


def spec(name="acme", impl="test_normalizer_impl:normalize", **kwargs):
    return normalizer.EntityNameNormalizerSpec(
        name=name,
        impl=impl,
        version=kwargs.pop("version", "1"),
        description=kwargs.pop("description", "Example normalizer"),
    )


def install_impl(monkeypatch, callback):
    monkeypatch.setitem(
        sys.modules, "test_normalizer_impl", SimpleNamespace(normalize=callback)
    )


def selected(monkeypatch, callback, name="acme"):
    install_impl(monkeypatch, callback)
    normalizer.register_entity_name_normalizer(spec(name))
    return normalizer.resolve_entity_name_normalizer(name)


def ep(name, register):
    return SimpleNamespace(
        name=name,
        value=f"{name}:register",
        dist=SimpleNamespace(name=f"dist-{name}", version="1"),
        load=lambda: register,
    )


# --- registration -----------------------------------------------------------


def test_registration_imports_no_implementation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("registration imported an implementation")

    monkeypatch.setattr(normalizer.importlib, "import_module", unexpected)
    normalizer.register_entity_name_normalizer(spec(), origin="package-a")
    normalizer.register_entity_name_normalizer(spec("another"), origin="package-b")
    assert normalizer.registered_entity_name_normalizer_names() == (
        "acme",
        "another",
    )


@pytest.mark.parametrize(
    "name",
    ["", "Upper", "a.b", "a:b", "-a", "_a", "a b", "a,b", "a=b", "a(b)", "a" * 65],
)
def test_reject_invalid_names(name):
    with pytest.raises(ValueError):
        normalizer.register_entity_name_normalizer(spec(name))


@pytest.mark.parametrize("name", ["0", "a_b-c", "a" * 64, "f", "c"])
def test_valid_names(name):
    normalizer.register_entity_name_normalizer(spec(name))
    assert name in normalizer.registered_entity_name_normalizer_names()


@pytest.mark.parametrize(
    "bad",
    [
        {"impl": ""},
        {"impl": "   "},
        {"version": 1},
        {"description": ""},
        {"description": "two\nlines"},
    ],
)
def test_reject_malformed_specs(bad):
    with pytest.raises(ValueError):
        normalizer.register_entity_name_normalizer(spec(**bad))


def test_duplicate_registration_is_reported_and_unselectable(lightrag_caplog):
    normalizer.register_entity_name_normalizer(spec(), origin="package-a")
    normalizer.register_entity_name_normalizer(spec(), origin="package-b")

    assert normalizer.selectable_entity_name_normalizer_names() == ()
    assert normalizer.registered_entity_name_normalizer_names() == ("acme",)
    messages = " ".join(r.getMessage() for r in lightrag_caplog.records)
    assert "duplicate normalizer 'acme'" in messages
    assert "package-a" in messages and "package-b" in messages
    with pytest.raises(ValueError, match="duplicate name 'acme'"):
        normalizer.resolve_entity_name_normalizer("acme")


# --- selection --------------------------------------------------------------


@pytest.mark.parametrize("value", ["pkg:func", "pkg.func", "/tmp/plugin", "pkg\\func"])
def test_operator_import_paths_have_actionable_error(value):
    with pytest.raises(ValueError, match="[Ii]mport paths.*entry.point"):
        normalizer.resolve_entity_name_normalizer(value)


def test_unknown_name_lists_available_names():
    normalizer.register_entity_name_normalizer(spec())
    with pytest.raises(ValueError, match="unknown.*missing.*acme"):
        normalizer.resolve_entity_name_normalizer("missing")


def test_invalid_selection_name_is_refused():
    with pytest.raises(ValueError, match="must match"):
        normalizer.resolve_entity_name_normalizer("Not A Name")


def test_unset_selects_nothing():
    assert normalizer.resolve_entity_name_normalizer("") is None
    assert normalizer.resolve_entity_name_normalizer(None) is None


def test_resolves_only_the_selected_implementation(monkeypatch):
    install_impl(monkeypatch, str.strip)
    normalizer.register_entity_name_normalizer(spec())
    normalizer.register_entity_name_normalizer(
        spec("broken", impl="does_not_exist:normalize")
    )

    bound = normalizer.resolve_entity_name_normalizer("acme")

    assert bound("  X ") == "X"
    assert (bound.spec.name, bound.spec.version) == ("acme", "1")
    with pytest.raises(ValueError, match="broken.*does_not_exist"):
        normalizer.resolve_entity_name_normalizer("broken")


@pytest.mark.parametrize("impl", ["nocolon", ":normalize", "test_normalizer_impl:"])
def test_malformed_implementation_reference_is_refused(impl):
    normalizer.register_entity_name_normalizer(spec(impl=impl))
    with pytest.raises(ValueError, match="cannot load"):
        normalizer.resolve_entity_name_normalizer("acme")


def test_not_callable_is_refused(monkeypatch):
    install_impl(monkeypatch, "not callable")
    normalizer.register_entity_name_normalizer(spec())
    with pytest.raises(ValueError, match="not callable"):
        normalizer.resolve_entity_name_normalizer("acme")


def test_wrong_arity_is_refused(monkeypatch):
    install_impl(monkeypatch, lambda a, b: a)
    normalizer.register_entity_name_normalizer(spec())
    with pytest.raises(ValueError, match="exactly one positional"):
        normalizer.resolve_entity_name_normalizer("acme")


def test_async_normalizer_is_refused(monkeypatch):
    async def normalize(name):
        return name

    install_impl(monkeypatch, normalize)
    normalizer.register_entity_name_normalizer(spec())
    with pytest.raises(ValueError, match="synchronous"):
        normalizer.resolve_entity_name_normalizer("acme")


# --- the call contract ------------------------------------------------------


def test_a_conforming_normalizer_is_applied(monkeypatch):
    bound = selected(monkeypatch, lambda name: name.lower())

    assert bound("118E Régiment") == "118e régiment"


def test_a_non_string_result_names_the_plugin(monkeypatch):
    bound = selected(monkeypatch, lambda name: None)

    with pytest.raises(normalizer.EntityNameNormalizerError, match="'acme'.*NoneType"):
        bound("x")


@pytest.mark.parametrize("empty", ["", "   "])
def test_an_empty_result_keeps_the_given_name(monkeypatch, empty, lightrag_caplog):
    bound = selected(monkeypatch, lambda name: empty)

    assert bound("Original") == "Original"
    assert any(
        "empty name" in r.getMessage() and "'acme'" in r.getMessage()
        for r in lightrag_caplog.records
    )


def test_a_result_over_the_character_limit_is_an_error(monkeypatch):
    too_long = "A" * (DEFAULT_ENTITY_NAME_MAX_LENGTH + 1)
    bound = selected(monkeypatch, lambda name: too_long)

    with pytest.raises(normalizer.EntityNameNormalizerError, match="'acme'.*limit"):
        bound("short")


def test_a_result_over_the_byte_limit_is_an_error(monkeypatch):
    # Within the character limit, over the byte limit: 3 bytes per character.
    chars = DEFAULT_ENTITY_NAME_MAX_BYTES // 3 + 1
    assert chars <= DEFAULT_ENTITY_NAME_MAX_LENGTH
    too_wide = "苹" * chars
    bound = selected(monkeypatch, lambda name: too_wide)

    with pytest.raises(normalizer.EntityNameNormalizerError, match="byte limit"):
        bound("short")


def test_a_result_at_the_limits_is_accepted(monkeypatch):
    at_limit = "A" * DEFAULT_ENTITY_NAME_MAX_LENGTH
    bound = selected(monkeypatch, lambda name: at_limit)

    assert bound("short") == at_limit


def test_a_result_is_never_truncated(monkeypatch):
    bound = selected(monkeypatch, lambda name: "B" * 400)

    with pytest.raises(normalizer.EntityNameNormalizerError):
        bound("x")


def test_a_non_idempotent_normalizer_is_an_error(monkeypatch):
    bound = selected(monkeypatch, lambda name: name + "!")

    with pytest.raises(normalizer.EntityNameNormalizerError, match="not idempotent"):
        bound("x")


def test_idempotence_is_judged_on_the_plugins_own_output(monkeypatch):
    # Stripping is idempotent; the second call must receive the first result.
    seen = []

    def normalize(name):
        seen.append(name)
        return name.strip()

    bound = selected(monkeypatch, normalize)

    assert bound("  X  ") == "X"
    assert seen == ["  X  ", "X"]


def test_a_plugin_exception_propagates_unchanged(monkeypatch):
    class Boom(Exception):
        pass

    def normalize(name):
        raise Boom("plugin bug")

    bound = selected(monkeypatch, normalize)

    with pytest.raises(Boom, match="plugin bug"):
        bound("x")


# --- discovery --------------------------------------------------------------


def test_discovery_registers_entry_points_without_importing_them(monkeypatch):
    def register():
        normalizer.register_entity_name_normalizer(spec())

    monkeypatch.setattr(
        normalizer, "entry_points", lambda **kwargs: [ep("acme", register)]
    )

    assert normalizer._discover() == ["acme"]
    assert normalizer.selectable_entity_name_normalizer_names() == ("acme",)
    assert normalizer._discover() == []  # once per process


def test_a_failing_provider_is_skipped_and_its_partial_registration_dropped(
    monkeypatch, lightrag_caplog
):
    def broken():
        normalizer.register_entity_name_normalizer(spec("half"))
        raise RuntimeError("provider bug")

    def fine():
        normalizer.register_entity_name_normalizer(spec("fine"))

    monkeypatch.setattr(
        normalizer,
        "entry_points",
        lambda **kwargs: [ep("broken", broken), ep("fine", fine)],
    )

    assert normalizer.load_and_resolve_entity_name_normalizer("") is None

    assert normalizer.registered_entity_name_normalizer_names() == ("fine",)
    messages = " ".join(r.getMessage() for r in lightrag_caplog.records)
    assert "skipped failed plugin 'broken'" in messages
    assert "provider bug" in messages


def test_a_selected_name_from_a_failed_provider_aborts_startup(
    monkeypatch, lightrag_caplog
):
    def broken():
        raise RuntimeError("provider bug")

    monkeypatch.setattr(
        normalizer, "entry_points", lambda **kwargs: [ep("acme", broken)]
    )

    with pytest.raises(ValueError, match="unknown normalizer 'acme'"):
        normalizer.load_and_resolve_entity_name_normalizer("acme")

    messages = " ".join(r.getMessage() for r in lightrag_caplog.records)
    assert "could not be activated; startup aborted" in messages
    assert "provider bug" in messages


def test_startup_resolves_a_discovered_normalizer(monkeypatch):
    install_impl(monkeypatch, str.strip)

    def register():
        normalizer.register_entity_name_normalizer(spec())

    monkeypatch.setattr(
        normalizer, "entry_points", lambda **kwargs: [ep("acme", register)]
    )

    bound = normalizer.load_and_resolve_entity_name_normalizer("acme")

    assert bound(" X ") == "X"


def test_the_selection_summary_names_what_is_registered_and_selected(
    monkeypatch, lightrag_caplog
):
    bound = selected(monkeypatch, str.strip)

    with lightrag_caplog.at_level("INFO", logger="lightrag"):
        normalizer.log_entity_name_normalizer_selection(bound)
        normalizer.log_entity_name_normalizer_selection(None)

    summaries = [
        r.getMessage()
        for r in lightrag_caplog.records
        if "registered=" in r.getMessage()
    ]
    assert "selected=acme" in summaries[0]
    assert "selected=(none); entity names are unchanged" in summaries[1]
