"""``normalize_entity_name`` applies the selected entity-name normalizer last.

Only for a complete name (``final_space_pass=True``): the extraction parsers pass
``False`` because they cap the name afterwards, and apply the normalizer
themselves on the capped key (see
tests/extraction/test_entity_name_normalizer_extraction.py).
"""

from __future__ import annotations

import pytest

from lightrag import entity_name_normalizer as normalizer
from lightrag.utils import normalize_entity_name

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def no_active_normalizer(monkeypatch):
    monkeypatch.setattr(normalizer, "_active", None)


def install(implementation):
    spec = normalizer.EntityNameNormalizerSpec("acme", "x:y", "1", "Test normalizer")
    normalizer.set_active_entity_name_normalizer(
        normalizer.EntityNameNormalizer(spec, implementation, "test")
    )


def test_without_a_normalizer_the_name_is_unchanged():
    assert normalize_entity_name("  118ème  RI ") == "118ème RI"


def test_a_complete_name_gets_the_normalizer_last():
    install(str.lower)

    assert normalize_entity_name("  118ÈME  RI ") == "118ème ri"


def test_the_normalizer_sees_the_name_after_the_built_in_rules():
    seen = []

    def record(name):
        seen.append(name)
        return name

    install(record)

    # A no-break space is a space for the built-in rules, before the plugin.
    normalize_entity_name("118e RI")

    assert seen[0] == "118e RI"


def test_a_name_that_is_not_complete_is_left_to_its_caller():
    install(str.lower)

    assert normalize_entity_name("ABC", final_space_pass=False) == "ABC"


def test_an_empty_name_never_reaches_the_normalizer():
    def fail(name):
        pytest.fail("the normalizer was called on an empty name")

    install(fail)

    assert normalize_entity_name("   ") == ""


def test_a_normalizer_that_breaks_its_contract_raises():
    install(lambda name: name + "!")

    with pytest.raises(normalizer.EntityNameNormalizerError, match="not idempotent"):
        normalize_entity_name("ABC")
