"""The selected entity-name normalizer runs after the cap, on every extraction path.

A step that shortens a name must not run before the length cap: two spellings
the cap cuts to one key would get different keys, and a rebuild then matches the
stored node with only one of its records and drops the facts of the other. The
normalizer therefore keys the CAPPED name, in the text and the JSON parser alike,
for entities and relation endpoints, and a normalizer that breaks its contract
fails the document instead of dropping the item.

The merge-only rule is in "Entity name normalization changes",
docs/design/PurgeRecoveryContract.md.
"""

from __future__ import annotations

import itertools
import json

import pytest

from lightrag import entity_name_normalizer as normalizer
from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH
from lightrag.operate import (
    _finalize_extracted_identifier,
    _process_extraction_result,
    _process_json_extraction_result,
    _truncate_entity_identifier,
)
from lightrag.utils import normalize_entity_name, remove_spaces_next_to_chinese

pytestmark = pytest.mark.offline


def _ordinal(name: str) -> str:
    """Shortens ``ème`` to ``e`` and lowercases: idempotent, and it moves the cut."""
    return name.replace("ème", "e").lower()


@pytest.fixture(autouse=True)
def no_active_normalizer(monkeypatch):
    monkeypatch.setattr(normalizer, "_active", None)


def install(implementation):
    spec = normalizer.EntityNameNormalizerSpec("acme", "x:y", "1", "Test normalizer")
    normalizer.set_active_entity_name_normalizer(
        normalizer.EntityNameNormalizer(spec, implementation, "test")
    )


async def _text_path(entities=(), relations=()):
    records = [
        f"entity<|#|>{name}<|#|>organization<|#|>A company." for name in entities
    ]
    records += [
        f"relation<|#|>{source}<|#|>{target}<|#|>supplies<|#|>One supplies the other."
        for source, target in relations
    ]
    return await _process_extraction_result(
        "\n".join(records) + "\n<|COMPLETE|>", "chunk-1", 1, "doc.txt"
    )


async def _json_path(entities=(), relations=()):
    payload = {
        "entities": [
            {"name": name, "type": "organization", "description": "A company."}
            for name in entities
        ],
        "relationships": [
            {
                "source": source,
                "target": target,
                "keywords": "supplies",
                "description": "One supplies the other.",
            }
            for source, target in relations
        ],
    }
    return await _process_json_extraction_result(
        json.dumps(payload, ensure_ascii=False), "chunk-1", 1, "doc.json"
    )


_PATHS = pytest.mark.parametrize(
    "parse", [_text_path, _json_path], ids=["text", "json"]
)


def _stored_key(raw: str) -> str:
    return _finalize_extracted_identifier(
        normalize_entity_name(raw, final_space_pass=False), "chunk-1", "Entity name"
    )


# --- selection and application ---------------------------------------------


def test_without_a_normalizer_names_are_unchanged():
    assert _stored_key("118ème Régiment d'Infanterie") == (
        "118ème Régiment d'Infanterie"
    )


@pytest.mark.asyncio
@_PATHS
async def test_without_a_normalizer_the_parsers_keep_both_spellings(parse):
    nodes, _edges = await parse(
        entities=["118ème Régiment d'Infanterie", "118e régiment d'infanterie"]
    )

    assert sorted(nodes) == [
        "118e régiment d'infanterie",
        "118ème Régiment d'Infanterie",
    ]


@pytest.mark.asyncio
@_PATHS
async def test_spellings_the_normalizer_equates_share_one_node(parse):
    install(_ordinal)

    nodes, _edges = await parse(
        entities=["118ème Régiment d'Infanterie", "118e régiment d'infanterie"]
    )

    assert list(nodes) == ["118e régiment d'infanterie"]
    assert len(nodes["118e régiment d'infanterie"]) == 2


@pytest.mark.asyncio
@_PATHS
async def test_relation_endpoints_get_the_same_key_as_the_entity(parse):
    install(_ordinal)

    nodes, edges = await parse(
        entities=["118ème Régiment d'Infanterie"],
        relations=[("118ème Régiment d'Infanterie", "2ème Bataillon")],
    )

    assert list(nodes) == ["118e régiment d'infanterie"]
    assert list(edges) == [("118e régiment d'infanterie", "2e bataillon")]
    assert edges[("118e régiment d'infanterie", "2e bataillon")][0]["src_id"] == (
        "118e régiment d'infanterie"
    )


@pytest.mark.asyncio
@_PATHS
async def test_endpoints_the_normalizer_equates_are_a_self_loop(parse):
    install(_ordinal)

    _nodes, edges = await parse(
        relations=[
            ("2ème Bataillon", "2e bataillon"),
            ("2ème Bataillon", "3e bataillon"),
        ]
    )

    assert list(edges) == [("2e bataillon", "3e bataillon")]


# --- the length cap ---------------------------------------------------------

_PREFIXES = [
    "A" * n
    for n in range(
        DEFAULT_ENTITY_NAME_MAX_LENGTH - 5, DEFAULT_ENTITY_NAME_MAX_LENGTH + 1
    )
]
# ``ème`` straddles the cut for these prefixes, and the normalizer shortens it.
_TAIL_PIECES = ["è", "m", "e", "X", " "]


def test_the_normalized_key_only_merges_names():
    new_key_by_old_key: dict[str, str] = {}
    for prefix in _PREFIXES:
        for length in range(1, 4):
            for pieces in itertools.product(_TAIL_PIECES, repeat=length):
                raw = prefix + "".join(pieces)
                normalizer.set_active_entity_name_normalizer(None)
                old_key = _stored_key(raw)
                install(_ordinal)
                new_key = _stored_key(raw)

                # Two spellings that shared a key must still share one.
                assert new_key_by_old_key.setdefault(old_key, new_key) == new_key, raw

    # ...and the normalizer did merge some: the check above is not vacuous.
    assert len(set(new_key_by_old_key.values())) < len(new_key_by_old_key)


def test_normalizing_before_the_cap_would_split_what_the_cap_makes_equal():
    first = "A" * (DEFAULT_ENTITY_NAME_MAX_LENGTH - 2) + "ème"
    second = "A" * (DEFAULT_ENTITY_NAME_MAX_LENGTH - 2) + "èmX"

    def cap(name: str) -> str:
        return remove_spaces_next_to_chinese(
            _truncate_entity_identifier(
                name, DEFAULT_ENTITY_NAME_MAX_LENGTH, "chunk-1", "Entity name"
            )
        )

    # The cap alone gives the two spellings one key.
    assert cap(first) == cap(second)
    # Normalizing first and capping after, the order this module rules out,
    # gives them two.
    assert cap(_ordinal(first)) != cap(_ordinal(second))
    # Capping first and normalizing after keeps them together.
    assert _ordinal(cap(first)) == _ordinal(cap(second))

    install(_ordinal)
    assert _stored_key(first) == _stored_key(second)


# --- a normalizer that breaks its contract fails the document ---------------


@pytest.mark.asyncio
@_PATHS
async def test_a_result_over_the_cap_fails_instead_of_being_cut(parse):
    install(lambda name: "A" * (DEFAULT_ENTITY_NAME_MAX_LENGTH + 1))

    with pytest.raises(normalizer.EntityNameNormalizerError, match="'acme'"):
        await parse(entities=["Short name"])


@pytest.mark.asyncio
@_PATHS
async def test_a_result_over_the_cap_on_a_relation_endpoint_fails(parse):
    install(lambda name: "A" * (DEFAULT_ENTITY_NAME_MAX_LENGTH + 1))

    with pytest.raises(normalizer.EntityNameNormalizerError):
        await parse(relations=[("Source", "Target")])


@pytest.mark.asyncio
@_PATHS
async def test_a_plugin_exception_fails_instead_of_dropping_the_item(parse):
    def broken(name):
        raise KeyError("plugin bug")

    install(broken)

    with pytest.raises(normalizer.EntityNameNormalizerError, match="KeyError"):
        await parse(entities=["Some name"])


@pytest.mark.asyncio
@_PATHS
async def test_a_non_idempotent_plugin_fails_instead_of_dropping_the_item(parse):
    install(lambda name: name + "!")

    with pytest.raises(normalizer.EntityNameNormalizerError, match="not idempotent"):
        await parse(entities=["Some name"])
