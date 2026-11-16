"""Third-party entity-name normalizers: registration, discovery and selection.

Mirrors the third-party chunker seam (``lightrag.chunker.registry`` and
``lightrag.chunker.plugins``): an installed package registers a lazy
``module:attribute`` reference under the ``lightrag.entity_name_normalizers``
entry-point group, and the operator selects it by registered name with
``CUSTOM_ENTITY_NAME_NORMALIZER``. Selection is a closed set, never an import
path. An invalid selection fails startup.

A normalizer is ``normalize(name: str) -> str`` and sees the name only: the
entity type is unknown for relation endpoints, and a result that depended on it
would give an entity and the relations that point at it different keys.

Contract for a normalizer, enforced here where it can be checked:

* it returns a ``str`` that fits the limits of any extracted name
  (``DEFAULT_ENTITY_NAME_MAX_LENGTH`` characters and
  ``DEFAULT_ENTITY_NAME_MAX_BYTES`` bytes). A longer result is an error and is
  never truncated again: a second cut would silently alter a canonical name;
* it is idempotent, ``normalize(normalize(x)) == normalize(x)``, because stored
  names pass through it again (graph API, rebuild after a deletion);
* an empty result keeps the name it was given, and is logged;
* an exception it raises is re-raised as :class:`EntityNameNormalizerError`,
  chained to the original. Callers on the extraction path wrap each item in a
  broad ``except Exception: continue``, which would drop the entity with only
  a warning; the typed error is the one they let through, so the document
  fails instead.

It must also be deterministic, free of side effects, and may only merge names
that used to differ, including from one version of the normalizer to the next.
That last property cannot be checked at run time.

**Where it runs.** The selected normalizer is process-wide, set once at
startup with :func:`set_active_entity_name_normalizer`, and applied through
:func:`apply_entity_name_normalizer`, the single application point. It runs
AFTER the length cap and the final space pass, never before: a step that
shortens a name, run before the cap, gives two spellings different keys where
the cap gave them one, and a rebuild then drops the facts of one of them.

This module logs through :mod:`logging` and imports nothing heavy so that
``lightrag.utils`` can use it without a circular import.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, Iterator

from lightrag.constants import (
    DEFAULT_ENTITY_NAME_MAX_BYTES,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)

logger = logging.getLogger("lightrag")

ENTRY_POINT_GROUP = "lightrag.entity_name_normalizers"
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class EntityNameNormalizerError(RuntimeError):
    """A selected normalizer broke its contract on a given name."""


@dataclass(frozen=True)
class EntityNameNormalizerSpec:
    """Metadata only; ``impl`` is a lazy ``module:attribute`` reference.

    ``version`` is an opaque author-supplied observation, never used to select
    or to admit a normalizer.
    """

    name: str
    impl: str
    version: str
    description: str


@dataclass(frozen=True)
class _Registration:
    spec: EntityNameNormalizerSpec
    origin: str


_REGISTRY: dict[str, _Registration] = {}
_DUPLICATES: set[str] = set()
_registration_origin: str | None = None
_loaded = False
_failures: list[tuple[str, str, str]] = []


@contextmanager
def _plugin_registration(origin: str) -> Iterator[None]:
    """Attribute registrations to their provider; discard partial failures."""
    global _registration_origin
    previous_origin = _registration_origin
    previous = _REGISTRY.copy()
    previous_duplicates = _DUPLICATES.copy()
    _registration_origin = origin
    try:
        yield
    except Exception:
        _REGISTRY.clear()
        _REGISTRY.update(previous)
        _DUPLICATES.clear()
        _DUPLICATES.update(previous_duplicates)
        raise
    finally:
        _registration_origin = previous_origin


def register_entity_name_normalizer(
    spec: EntityNameNormalizerSpec, *, origin: str | None = None
) -> None:
    """Register without importing an implementation; report every collision."""
    if not isinstance(spec.name, str) or not _NAME.fullmatch(spec.name):
        raise ValueError(
            "entity-name normalizer name must match [a-z0-9][a-z0-9_-]{0,63}"
        )
    if not isinstance(spec.impl, str) or not spec.impl.strip():
        raise ValueError(
            f"entity-name normalizer {spec.name!r} requires a lazy "
            "implementation reference"
        )
    if not isinstance(spec.version, str):
        raise ValueError(
            f"entity-name normalizer {spec.name!r} version must be a string"
        )
    if (
        not isinstance(spec.description, str)
        or not spec.description.strip()
        or any(c in spec.description for c in "\r\n")
    ):
        raise ValueError(
            f"entity-name normalizer {spec.name!r} requires a one-line description"
        )
    resolved_origin = _registration_origin or origin or spec.impl
    previous = _REGISTRY.get(spec.name)
    if previous is not None:
        _DUPLICATES.add(spec.name)
        logger.error(
            "[entity-name-normalizers] duplicate normalizer %r: %s -> %s",
            spec.name,
            previous.origin,
            resolved_origin,
        )
    _REGISTRY[spec.name] = _Registration(spec, resolved_origin)


def registered_entity_name_normalizer_names() -> tuple[str, ...]:
    """Every registered name, including names that collided."""
    return tuple(sorted(_REGISTRY))


def selectable_entity_name_normalizer_names() -> tuple[str, ...]:
    """The names ``CUSTOM_ENTITY_NAME_NORMALIZER`` can actually be set to.

    A duplicated name is registered but unselectable: resolution refuses it.
    """
    return tuple(sorted(name for name in _REGISTRY if name not in _DUPLICATES))


class EntityNameNormalizer:
    """A resolved normalizer: the implementation plus its provenance.

    Calling it applies the implementation and checks the contract documented
    in the module docstring. An exception raised by the implementation is
    re-raised as :class:`EntityNameNormalizerError`; it is never replaced by
    the name it was given.
    """

    def __init__(
        self,
        spec: EntityNameNormalizerSpec,
        implementation: Callable[[str], str],
        origin: str,
    ) -> None:
        self.spec = spec
        self.implementation = implementation
        self.origin = origin

    def _describe(self) -> str:
        return (
            f"entity-name normalizer {self.spec.name!r} "
            f"(version {self.spec.version}, {self.origin})"
        )

    def _apply(self, name: str) -> str:
        try:
            return self.implementation(name)
        except Exception as exc:
            raise EntityNameNormalizerError(
                f"{self._describe()} raised {type(exc).__name__} for {name!r}: {exc}"
            ) from exc

    def __call__(self, name: str) -> str:
        result = self._apply(name)
        if not isinstance(result, str):
            raise EntityNameNormalizerError(
                f"{self._describe()} returned {type(result).__name__}, not str, "
                f"for {name!r}"
            )
        if not result.strip():
            logger.warning(
                "%s returned an empty name for %r; keeping the name it was given",
                self._describe(),
                name,
            )
            return name
        if (
            len(result) > DEFAULT_ENTITY_NAME_MAX_LENGTH
            or len(result.encode("utf-8")) > DEFAULT_ENTITY_NAME_MAX_BYTES
        ):
            raise EntityNameNormalizerError(
                f"{self._describe()} returned a name of {len(result)} characters / "
                f"{len(result.encode('utf-8'))} bytes for {name!r}, over the "
                f"{DEFAULT_ENTITY_NAME_MAX_LENGTH} character / "
                f"{DEFAULT_ENTITY_NAME_MAX_BYTES} byte limit; the result is not "
                "truncated"
            )
        again = self._apply(result)
        if again != result:
            raise EntityNameNormalizerError(
                f"{self._describe()} is not idempotent: it maps {name!r} to "
                f"{result!r} and then {result!r} to {again!r}"
            )
        return result


_active: EntityNameNormalizer | None = None


def set_active_entity_name_normalizer(
    normalizer: EntityNameNormalizer | None,
) -> None:
    """Make ``normalizer`` the process-wide one; ``None`` turns it off.

    Called once at startup, before any entity name is normalized. Changing it
    on a workspace that already holds entities is a change of normalizer, with
    the consequences described in the entity name normalization section of
    ``docs/design/PurgeRecoveryContract.md``.
    """
    global _active
    _active = normalizer


def get_active_entity_name_normalizer() -> EntityNameNormalizer | None:
    """The process-wide normalizer, or ``None`` when none is selected."""
    return _active


def apply_entity_name_normalizer(name: str) -> str:
    """The single application point: ``name`` as the active normalizer keys it.

    Returns ``name`` unchanged when no normalizer is active or the name is
    empty. Callers must pass a name that has already been capped and put
    through the built-in rules; see the module docstring.
    """
    normalizer = _active
    if normalizer is None or not name.strip():
        return name
    return normalizer(name)


def resolve_entity_name_normalizer(name: str | None) -> EntityNameNormalizer | None:
    """Fail startup on an invalid selection or import; unset returns ``None``."""
    if name is None or name == "":
        return None
    if not isinstance(name, str):
        raise ValueError(
            "CUSTOM_ENTITY_NAME_NORMALIZER must be a registered normalizer name"
        )
    if any(c in name for c in ":/\\."):
        raise ValueError(
            "CUSTOM_ENTITY_NAME_NORMALIZER: import paths are not supported; "
            "install a plugin using lightrag.entity_name_normalizers entry "
            "points and select its registered name"
        )
    if not _NAME.fullmatch(name):
        raise ValueError(
            "CUSTOM_ENTITY_NAME_NORMALIZER must match [a-z0-9][a-z0-9_-]{0,63}"
        )
    if name in _DUPLICATES:
        raise ValueError(
            f"CUSTOM_ENTITY_NAME_NORMALIZER selects duplicate name {name!r}; "
            "remove the conflicting registration (see both origins in the "
            "error log)"
        )
    registration = _REGISTRY.get(name)
    if registration is None:
        raise ValueError(
            f"CUSTOM_ENTITY_NAME_NORMALIZER: unknown normalizer {name!r}; "
            f"selectable names: "
            f"{', '.join(selectable_entity_name_normalizer_names()) or '(none)'}. "
            "Check plugin discovery errors for a failed provider."
        )
    spec = registration.spec
    try:
        module, separator, attribute = spec.impl.partition(":")
        if not module or not separator or not attribute:
            raise ValueError("expected module:attribute")
        implementation = importlib.import_module(module)
        for part in attribute.split("."):
            implementation = getattr(implementation, part)
    except Exception as exc:
        raise ValueError(
            f"cannot load selected entity-name normalizer {name!r} "
            f"({spec.impl}, {registration.origin}): {exc}"
        ) from exc
    if not callable(implementation):
        raise ValueError(
            f"selected entity-name normalizer {name!r} ({spec.impl}) is not callable"
        )
    if inspect.iscoroutinefunction(implementation) or inspect.iscoroutinefunction(
        getattr(implementation, "__call__", None)
    ):
        raise ValueError(
            f"selected entity-name normalizer {name!r} must be synchronous: "
            "entity names are normalized from synchronous code"
        )
    try:
        signature = inspect.signature(implementation)
    except (ValueError, TypeError):
        pass  # Some native callables expose no signature; do not invent a fence.
    else:
        try:
            signature.bind(None)
        except TypeError as exc:
            raise ValueError(
                f"selected entity-name normalizer {name!r} must accept exactly "
                f"one positional argument, the name: {exc}"
            ) from exc
    return EntityNameNormalizer(spec, implementation, registration.origin)


def _origin(ep) -> str:
    dist = getattr(ep, "dist", None)
    return (
        f"{getattr(dist, 'name', 'unknown distribution')}"
        f"@{getattr(dist, 'version', '?')} {ep.name}={ep.value}"
    )


def _discover() -> list[str]:
    """Cache registrations and failure diagnostics, never implementations."""
    global _loaded
    if _loaded:
        return []
    _loaded = True
    loaded = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=_origin):
        origin = _origin(ep)
        try:
            with _plugin_registration(origin):
                register = ep.load()
                register()
        except Exception as exc:
            _failures.append((ep.name, origin, str(exc)))
            continue
        loaded.append(ep.name)
    return loaded


def _report_failures(outcome: str) -> None:
    for name, origin, error in _failures:
        logger.error(
            "[entity-name-normalizers] skipped failed plugin %r (%s): %s; %s",
            name,
            origin,
            error,
            outcome,
        )


def load_and_resolve_entity_name_normalizer(
    name: str | None,
) -> EntityNameNormalizer | None:
    """Discover installed providers once, then resolve the operator's selection.

    A failed provider is logged and skipped, but an invalid selection aborts
    startup, and the log says which provider failures may be the cause.
    """
    _discover()
    try:
        selected = resolve_entity_name_normalizer(name)
    except ValueError:
        _report_failures(
            f"CUSTOM_ENTITY_NAME_NORMALIZER={name!r} could not be activated; "
            "startup aborted"
        )
        raise
    _report_failures(
        f"selected normalizer {selected.spec.name!r} validated"
        if selected
        else "CUSTOM_ENTITY_NAME_NORMALIZER is unset; entity names are unchanged"
    )
    return selected


def log_entity_name_normalizer_selection(
    selected: EntityNameNormalizer | None,
) -> None:
    """One startup summary of what is registered and what is selected."""
    registered = [
        {
            "name": r.spec.name,
            "version": r.spec.version,
            "origin": r.origin,
            "description": r.spec.description,
        }
        for _, r in sorted(_REGISTRY.items())
    ]
    logger.info(
        "[entity-name-normalizers] registered=%s; selected=%s%s",
        registered,
        selected.spec.name if selected else "(none)",
        "" if selected else "; entity names are unchanged",
    )
