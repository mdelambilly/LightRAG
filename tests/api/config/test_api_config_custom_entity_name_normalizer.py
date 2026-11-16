"""Operator selection is a name, with CLI taking precedence over .env."""

import sys

import pytest

from lightrag.api.config import parse_args

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("value", ["", "archives-normalizer"])
def test_custom_entity_name_normalizer_env(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", value)
    assert parse_args().custom_entity_name_normalizer == value


def test_custom_entity_name_normalizer_defaults_to_unset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.delenv("CUSTOM_ENTITY_NAME_NORMALIZER", raising=False)
    assert parse_args().custom_entity_name_normalizer == ""


def test_custom_entity_name_normalizer_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "env-normalizer")
    monkeypatch.setattr(
        sys,
        "argv",
        ["lightrag-server", "--custom-entity-name-normalizer", "cli-normalizer"],
    )
    assert parse_args().custom_entity_name_normalizer == "cli-normalizer"
