"""Tests for enum mapping on slots with multiple enum ranges via any_of.

Verifies that when a source slot uses ``any_of`` to specify multiple enum
ranges, the transformer correctly iterates enum derivations in order and
maps permissible values across all enums.

See: https://github.com/linkml/linkml-map/issues/146
     https://github.com/linkml/linkml/issues/2128
"""

import copy

import pytest
from linkml_runtime import SchemaView

from linkml_map.transformer.errors import TransformationError
from linkml_map.transformer.object_transformer import ObjectTransformer

SOURCE_SCHEMA = """\
id: https://example.org/lights-source
name: lights-source
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  PrimaryColors:
    permissible_values:
      light_red:
      dark_red:
      light_green:
      dark_green:
      light_blue:
      dark_blue:
  SecondaryColors:
    permissible_values:
      light_cyan:
      dark_cyan:
      light_magenta:
      dark_magenta:
  Missingness:
    permissible_values:
      not_available:
      other:
classes:
  Light:
    attributes:
      id:
        identifier: true
        range: string
      color:
        any_of:
          - range: PrimaryColors
          - range: SecondaryColors
          - range: Missingness
      colors:
        multivalued: true
        any_of:
          - range: PrimaryColors
          - range: SecondaryColors
          - range: Missingness
  Container:
    tree_root: true
    attributes:
      lights:
        range: Light
        multivalued: true
        inlined_as_list: true
"""

TARGET_SCHEMA = """\
id: https://example.org/lights-target
name: lights-target
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  SimplePrimary:
    permissible_values:
      red:
      green:
      blue:
  SimpleSecondary:
    permissible_values:
      cyan:
      magenta:
  MissingnessTarget:
    permissible_values:
      na:
      oth:
classes:
  Light:
    attributes:
      id:
        identifier: true
        range: string
      color:
        any_of:
          - range: SimplePrimary
          - range: SimpleSecondary
          - range: MissingnessTarget
      colors:
        multivalued: true
        any_of:
          - range: SimplePrimary
          - range: SimpleSecondary
          - range: MissingnessTarget
  Container:
    tree_root: true
    attributes:
      lights:
        range: Light
        multivalued: true
        inlined_as_list: true
"""

TRANSFORM_SPEC = {
    "class_derivations": {
        "Light": {
            "populated_from": "Light",
            "slot_derivations": {
                "id": {},
                "color": {"populated_from": "color"},
                "colors": {"populated_from": "colors"},
            },
        },
        "Container": {
            "populated_from": "Container",
            "slot_derivations": {
                "lights": {"populated_from": "lights"},
            },
        },
    },
    "enum_derivations": {
        "SimplePrimary": {
            "name": "SimplePrimary",
            "populated_from": "PrimaryColors",
            "permissible_value_derivations": {
                "red": {
                    "name": "red",
                    "sources": ["light_red", "dark_red"],
                },
                "green": {
                    "name": "green",
                    "sources": ["light_green", "dark_green"],
                },
                "blue": {
                    "name": "blue",
                    "sources": ["light_blue", "dark_blue"],
                },
            },
        },
        "SimpleSecondary": {
            "name": "SimpleSecondary",
            "populated_from": "SecondaryColors",
            "permissible_value_derivations": {
                "cyan": {
                    "name": "cyan",
                    "sources": ["light_cyan", "dark_cyan"],
                },
                "magenta": {
                    "name": "magenta",
                    "sources": ["light_magenta", "dark_magenta"],
                },
            },
        },
        "MissingnessTarget": {
            "name": "MissingnessTarget",
            "populated_from": "Missingness",
            "permissible_value_derivations": {
                "na": {
                    "name": "na",
                    "populated_from": "not_available",
                },
                "oth": {
                    "name": "oth",
                    "populated_from": "other",
                },
            },
        },
    },
}


def _make_transformer() -> ObjectTransformer:
    """Build an ObjectTransformer wired to the source/target schemas and spec."""
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)
    tr.create_transformer_specification(copy.deepcopy(TRANSFORM_SPEC))
    return tr


@pytest.mark.parametrize(
    "source_color,expected",
    [
        ("light_red", "red"),
        ("dark_red", "red"),
        ("light_green", "green"),
        ("dark_blue", "blue"),
        ("light_cyan", "cyan"),
        ("dark_magenta", "magenta"),
        ("not_available", "na"),
        ("other", "oth"),
    ],
    ids=[
        "primary-light_red",
        "primary-dark_red",
        "primary-light_green",
        "primary-dark_blue",
        "secondary-light_cyan",
        "secondary-dark_magenta",
        "missingness-not_available",
        "missingness-other",
    ],
)
def test_single_valued_multi_enum(source_color, expected):
    """Single-valued slot with any_of enum ranges maps correctly."""
    tr = _make_transformer()
    source = {"id": "light1", "color": source_color}
    result = tr.map_object(source, source_type="Light")
    assert result["color"] == expected


def test_multivalued_multi_enum():
    """Multivalued slot with any_of enum ranges maps all values."""
    tr = _make_transformer()
    source = {
        "id": "light1",
        "colors": ["light_red", "dark_green", "light_magenta", "not_available"],
    }
    result = tr.map_object(source, source_type="Light")
    assert result["colors"] == ["red", "green", "magenta", "na"]


def test_no_matching_enum_returns_none():
    """Value not in any enum derivation returns None."""
    tr = _make_transformer()
    source = {"id": "light1", "color": "nonexistent_value"}
    result = tr.map_object(source, source_type="Light")
    assert result["color"] is None


def test_container_with_multi_enum():
    """End-to-end: container with nested objects using multi-enum slots."""
    tr = _make_transformer()
    source = {
        "lights": [
            {
                "id": "l1",
                "color": "light_red",
                "colors": ["light_red", "dark_green", "not_available"],
            },
            {
                "id": "l2",
                "color": "light_cyan",
                "colors": ["light_magenta"],
            },
        ],
    }
    result = tr.map_object(source, source_type="Container")
    assert result["lights"][0]["color"] == "red"
    assert result["lights"][0]["colors"] == ["red", "green", "na"]
    assert result["lights"][1]["color"] == "cyan"
    assert result["lights"][1]["colors"] == ["magenta"]


def test_mirror_source_stops_iteration():
    """mirror_source on an earlier enum prevents trying later enums."""
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)

    spec = copy.deepcopy(TRANSFORM_SPEC)
    # Set mirror_source on PrimaryColors derivation
    spec["enum_derivations"]["SimplePrimary"]["mirror_source"] = True
    tr.create_transformer_specification(spec)

    # "unknown_value" doesn't match any PrimaryColors PV derivation,
    # but mirror_source=True means it returns unchanged
    source = {"id": "light1", "color": "unknown_value"}
    result = tr.map_object(source, source_type="Light")
    assert result["color"] == "unknown_value"


def test_null_color_stays_none():
    """Null source value is not transformed."""
    tr = _make_transformer()
    source = {"id": "light1", "color": None}
    result = tr.map_object(source, source_type="Light")
    assert result["color"] is None


def test_pv_populated_from_list_form_maps_multiple_sources():
    """populated_from as a list maps any listed source value to the target PV."""
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)
    spec = copy.deepcopy(TRANSFORM_SPEC)
    # Replace SimplePrimary's deprecated sources with list-form populated_from.
    spec["enum_derivations"]["SimplePrimary"]["permissible_value_derivations"] = {
        "red": {"populated_from": ["light_red", "dark_red"]},
        "green": {"populated_from": ["light_green", "dark_green"]},
        "blue": {"populated_from": ["light_blue", "dark_blue"]},
    }
    tr.create_transformer_specification(spec)

    for src, expected in [("light_red", "red"), ("dark_red", "red"), ("dark_blue", "blue")]:
        result = tr.map_object({"id": "l1", "color": src}, source_type="Light")
        assert result["color"] == expected


def test_pv_populated_from_scalar_form_still_works():
    """Scalar populated_from is wrapped to a one-element list at load time."""
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)
    spec = copy.deepcopy(TRANSFORM_SPEC)
    spec["enum_derivations"]["MissingnessTarget"]["permissible_value_derivations"] = {
        "na": {"populated_from": "not_available"},
        "oth": {"populated_from": "other"},
    }
    tr.create_transformer_specification(spec)
    # Pydantic representation is always a list after normalization.
    pvds = tr.specification.enum_derivations["MissingnessTarget"].permissible_value_derivations
    assert pvds["na"].populated_from == ["not_available"]

    result = tr.map_object({"id": "l1", "color": "not_available"}, source_type="Light")
    assert result["color"] == "na"


def test_pv_sources_only_is_migrated_to_populated_from():
    """sources-only PV derivs are migrated so the runtime sees only populated_from."""
    tr = _make_transformer()
    # SimplePrimary in TRANSFORM_SPEC uses sources: [...] — confirm migration cleared sources.
    pvds = tr.specification.enum_derivations["SimplePrimary"].permissible_value_derivations
    assert pvds["red"].populated_from == ["light_red", "dark_red"]
    # sources is cleared after migration — the runtime relies on populated_from alone.
    assert not pvds["red"].sources


def _make_transformer_with_enum_expr(expr: str, unrestricted_eval: bool) -> ObjectTransformer:
    """Build a transformer whose SimplePrimary enum derivation carries *expr*."""
    tr = ObjectTransformer(unrestricted_eval=unrestricted_eval)
    tr.source_schemaview = SchemaView(SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)
    spec = copy.deepcopy(TRANSFORM_SPEC)
    spec["enum_derivations"]["SimplePrimary"]["expr"] = expr
    tr.create_transformer_specification(spec)
    return tr


def test_enum_expr_restricted_evaluates():
    """A restricted-syntax enum expr is evaluated against the source object, no opt-in needed."""
    tr = _make_transformer_with_enum_expr("'EXPR:' + color", unrestricted_eval=False)
    result = tr.map_object({"id": "light1", "color": "light_red"}, source_type="Light")
    assert result["color"] == "EXPR:light_red"


def test_enum_expr_unrestricted_fallback():
    """An expression rejected by simpleeval falls back to asteval only when unrestricted_eval is opted in."""
    tr = _make_transformer_with_enum_expr("target = src['color']", unrestricted_eval=True)
    result = tr.map_object({"id": "light1", "color": "light_red"}, source_type="Light")
    assert result["color"] == "light_red"


def test_enum_expr_restricted_raises_without_opt_in():
    """The same asteval-only expr raises rather than silently escalating when not opted in."""
    tr = _make_transformer_with_enum_expr("target = src['color']", unrestricted_eval=False)
    with pytest.raises(TransformationError):
        tr.map_object({"id": "light1", "color": "light_red"}, source_type="Light")


@pytest.mark.parametrize("unrestricted_eval", [False, True])
def test_enum_expr_strict_unknown_name_raises_without_fallback(unrestricted_eval):
    """A strict-mode typo in an enum expr raises and never escalates to unrestricted eval."""
    tr = ObjectTransformer(strict=True, unrestricted_eval=unrestricted_eval)
    tr.source_schemaview = SchemaView(SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)
    spec = copy.deepcopy(TRANSFORM_SPEC)
    spec["enum_derivations"]["SimplePrimary"]["expr"] = "colorr"
    tr.create_transformer_specification(spec)
    with pytest.raises(TransformationError):
        tr.map_object({"id": "light1", "color": "light_red"}, source_type="Light")


def test_scalar_enum_expr_can_reference_source_as_src_in_restricted_mode():
    """A plain enum-range slot passes the scalar value; the expr sees it as ``src`` without opt-in."""
    source_schema = """\
id: https://example.org/scalar-source
name: scalar-source
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  Color:
    permissible_values:
      red:
      green:
classes:
  Rec:
    tree_root: true
    attributes:
      id:
        identifier: true
        range: string
      color:
        range: Color
"""
    target_schema = source_schema.replace("scalar-source", "scalar-target").replace("Color", "TColor")

    tr = ObjectTransformer(unrestricted_eval=False)
    tr.source_schemaview = SchemaView(source_schema)
    tr.target_schemaview = SchemaView(target_schema)
    tr.create_transformer_specification(
        {
            "class_derivations": {
                "Rec": {
                    "populated_from": "Rec",
                    "slot_derivations": {"id": {}, "color": {"populated_from": "color"}},
                }
            },
            "enum_derivations": {
                "TColor": {"name": "TColor", "populated_from": "Color", "expr": "'X:' + src"},
            },
        }
    )

    result = tr.map_object({"id": "r1", "color": "red"}, source_type="Rec")
    assert result["color"] == "X:red"


def test_explicit_range_any_with_any_of():
    """Slots with explicit range: Any plus any_of enum ranges are mapped correctly."""
    schema = SOURCE_SCHEMA.replace(
        "      color:\n        any_of:",
        "      color:\n        range: Any\n        any_of:",
    )
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(schema)
    tr.target_schemaview = SchemaView(TARGET_SCHEMA)
    tr.create_transformer_specification(copy.deepcopy(TRANSFORM_SPEC))

    source = {"id": "light1", "color": "light_red"}
    result = tr.map_object(source, source_type="Light")
    assert result["color"] == "red"


# ---------------------------------------------------------------------------
# mirror_source with null source values
#
# ``mirror_source`` falls back to ``str(source_value)`` when no permissible
# value derivation matched, so nulls must be filtered out first or they become
# the literal strings "None" / "nan". A null scalar slot is short-circuited by
# ObjectTransformer._derive_slot before it ever reaches transform_enum, but
# NaN scalars and null *elements* of a multivalued slot do reach it, as does a
# bare enum value passed straight to map_object (the session/CLI entry point).
# ---------------------------------------------------------------------------

NULL_MIRROR_SOURCE_SCHEMA = """\
id: https://example.org/null-mirror-source
name: null-mirror-source
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  SrcColors:
    permissible_values:
      red_src:
      blue_src:
classes:
  Rec:
    tree_root: true
    attributes:
      id:
        identifier: true
        range: string
      color_any_of:
        any_of:
          - range: SrcColors
      colors_any_of:
        multivalued: true
        any_of:
          - range: SrcColors
      color_ranged:
        range: SrcColors
      colors_ranged:
        range: SrcColors
        multivalued: true
"""

NULL_MIRROR_TARGET_SCHEMA = """\
id: https://example.org/null-mirror-target
name: null-mirror-target
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  TgtColors:
    permissible_values:
      red:
      blue:
classes:
  Rec:
    tree_root: true
    attributes:
      id:
        identifier: true
        range: string
      color_any_of:
        any_of:
          - range: TgtColors
      colors_any_of:
        multivalued: true
        any_of:
          - range: TgtColors
      color_ranged:
        range: TgtColors
      colors_ranged:
        range: TgtColors
        multivalued: true
"""

NULL_MIRROR_TRANSFORM_SPEC = {
    "class_derivations": {
        "Rec": {
            "populated_from": "Rec",
            "slot_derivations": {
                "id": {},
                "color_any_of": {"populated_from": "color_any_of"},
                "colors_any_of": {"populated_from": "colors_any_of"},
                "color_ranged": {"populated_from": "color_ranged"},
                "colors_ranged": {"populated_from": "colors_ranged"},
            },
        },
    },
    "enum_derivations": {
        "TgtColors": {
            "name": "TgtColors",
            "populated_from": "SrcColors",
            "mirror_source": True,
            "permissible_value_derivations": {
                "red": {"name": "red", "populated_from": "red_src"},
            },
        },
    },
}

# Null values that must never be mirrored as strings. NaN is what a delimited
# file loaded through pandas yields for an empty cell.
NULLISH = [
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
]


def _make_null_mirror_transformer() -> ObjectTransformer:
    """Build a transformer whose only enum derivation sets ``mirror_source: true``."""
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(NULL_MIRROR_SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(NULL_MIRROR_TARGET_SCHEMA)
    tr.create_transformer_specification(copy.deepcopy(NULL_MIRROR_TRANSFORM_SPEC))
    return tr


@pytest.mark.parametrize("slot", ["color_any_of", "color_ranged"])
@pytest.mark.parametrize("null_value", NULLISH)
def test_mirror_source_null_scalar_is_not_stringified(slot: str, null_value: float | None) -> None:
    """A null scalar under ``mirror_source`` yields None, not "None"/"nan"."""
    tr = _make_null_mirror_transformer()
    result = tr.map_object({"id": "r1", slot: null_value}, source_type="Rec")
    assert result[slot] is None


@pytest.mark.parametrize("slot", ["colors_any_of", "colors_ranged"])
@pytest.mark.parametrize("null_value", NULLISH)
def test_mirror_source_null_list_element_is_not_stringified(slot: str, null_value: float | None) -> None:
    """Null elements of a multivalued slot reach transform_enum and mirror to None.

    The surrounding values still map (``red_src`` via its derivation) or mirror
    (``blue_src``, which has no derivation), so the null handling is not just
    nulling the whole list.
    """
    tr = _make_null_mirror_transformer()
    result = tr.map_object({"id": "r1", slot: ["red_src", null_value, "blue_src"]}, source_type="Rec")
    assert result[slot] == ["red", None, "blue_src"]


@pytest.mark.parametrize("null_value", NULLISH)
def test_mirror_source_null_bare_enum_value(null_value: float | None) -> None:
    """Mapping a bare enum value - the session/CLI entry point - mirrors nulls to None."""
    tr = _make_null_mirror_transformer()
    assert tr.map_object(null_value, source_type="SrcColors", target_type="TgtColors") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        (float("nan"), True),
        (float("inf"), False),
        (0, False),
        (0.0, False),
        ("", False),
        ("nan", False),
        ("red_src", False),
        ([], False),
    ],
)
def test_is_none_or_nan(value: object, expected: bool) -> None:
    """Only None and float NaN count as null; falsy values and the string "nan" do not."""
    assert ObjectTransformer._is_none_or_nan(value) is expected


# ---------------------------------------------------------------------------
# any_of mixing enum and non-enum ranges
#
# A slot may list both enum and non-enum ranges under any_of. Values matching
# an enum derivation map as enums; whatever is left over is retried against the
# non-enum range(s). A slot whose any_of lists enums only has nothing to fall
# back to, so an unmatched value becomes None.
#
# See ObjectTransformer._map_value_by_range / _get_any_of_nonenum_names.
# ---------------------------------------------------------------------------

MIXED_ANY_OF_SOURCE_SCHEMA = """\
id: https://example.org/mixed-any-of-source
name: mixed-any-of-source
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  SrcPrimary:
    permissible_values:
      light_red:
      dark_red:
  SrcSecondary:
    permissible_values:
      light_cyan:
      dark_cyan:
classes:
  Rec:
    tree_root: true
    attributes:
      id:
        identifier: true
        range: string
      mixed:
        any_of:
          - range: SrcPrimary
          - range: SrcSecondary
          - range: string
      mixed_multi:
        multivalued: true
        any_of:
          - range: SrcPrimary
          - range: SrcSecondary
          - range: string
      enums_only:
        any_of:
          - range: SrcPrimary
          - range: SrcSecondary
      enums_only_multi:
        multivalued: true
        any_of:
          - range: SrcPrimary
          - range: SrcSecondary
      mixed_int:
        any_of:
          - range: SrcPrimary
          - range: integer
      plain_strings:
        range: string
        multivalued: true
      plain_ints:
        range: integer
        multivalued: true
"""

MIXED_ANY_OF_TARGET_SCHEMA = """\
id: https://example.org/mixed-any-of-target
name: mixed-any-of-target
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
enums:
  TgtPrimary:
    permissible_values:
      red:
  TgtSecondary:
    permissible_values:
      cyan:
classes:
  Rec:
    tree_root: true
    attributes:
      id:
        identifier: true
        range: string
      mixed:
        any_of:
          - range: TgtPrimary
          - range: TgtSecondary
          - range: string
      mixed_multi:
        multivalued: true
        any_of:
          - range: TgtPrimary
          - range: TgtSecondary
          - range: string
      enums_only:
        any_of:
          - range: TgtPrimary
          - range: TgtSecondary
      enums_only_multi:
        multivalued: true
        any_of:
          - range: TgtPrimary
          - range: TgtSecondary
      mixed_int:
        any_of:
          - range: TgtPrimary
          - range: integer
      plain_strings:
        range: string
        multivalued: true
      plain_ints:
        range: integer
        multivalued: true
"""

MIXED_ANY_OF_TRANSFORM_SPEC = {
    "class_derivations": {
        "Rec": {
            "populated_from": "Rec",
            "slot_derivations": {
                "id": {},
                "mixed": {"populated_from": "mixed"},
                "mixed_multi": {"populated_from": "mixed_multi"},
                "enums_only": {"populated_from": "enums_only"},
                "enums_only_multi": {"populated_from": "enums_only_multi"},
                "mixed_int": {"populated_from": "mixed_int"},
                "plain_strings": {"populated_from": "plain_strings"},
                "plain_ints": {"populated_from": "plain_ints"},
            },
        },
    },
    "enum_derivations": {
        "TgtPrimary": {
            "name": "TgtPrimary",
            "populated_from": "SrcPrimary",
            "permissible_value_derivations": {
                "red": {"name": "red", "populated_from": ["light_red", "dark_red"]},
            },
        },
        "TgtSecondary": {
            "name": "TgtSecondary",
            "populated_from": "SrcSecondary",
            "permissible_value_derivations": {
                "cyan": {"name": "cyan", "populated_from": ["light_cyan", "dark_cyan"]},
            },
        },
    },
}


def _make_mixed_any_of_transformer(slot_ranges: dict[str, str] | None = None) -> ObjectTransformer:
    """Build a transformer for slots whose any_of mixes enum and non-enum ranges.

    :param slot_ranges: optional ``{slot name: range}`` declared on the slot
        derivations, which switches on the target-range coercion step.
    """
    tr = ObjectTransformer()
    tr.source_schemaview = SchemaView(MIXED_ANY_OF_SOURCE_SCHEMA)
    tr.target_schemaview = SchemaView(MIXED_ANY_OF_TARGET_SCHEMA)
    spec = copy.deepcopy(MIXED_ANY_OF_TRANSFORM_SPEC)
    for slot_name, rng in (slot_ranges or {}).items():
        spec["class_derivations"]["Rec"]["slot_derivations"][slot_name]["range"] = rng
    tr.create_transformer_specification(spec)
    return tr


@pytest.mark.parametrize("slot", ["mixed", "enums_only"])
@pytest.mark.parametrize(
    ("source_value", "expected"),
    [
        ("light_red", "red"),
        ("dark_red", "red"),
        ("light_cyan", "cyan"),
        ("dark_cyan", "cyan"),
    ],
)
def test_mixed_any_of_scalar_prefers_enum_mapping(slot: str, source_value: str, expected: str) -> None:
    """A value matching any enum derivation maps as an enum, non-enum range present or not."""
    tr = _make_mixed_any_of_transformer()
    result = tr.map_object({"id": "r1", slot: source_value}, source_type="Rec")
    assert result[slot] == expected


@pytest.mark.parametrize("source_value", ["free text", "light_blue", "42"])
def test_mixed_any_of_scalar_falls_back_to_non_enum_range(source_value: str) -> None:
    """With no enum mapping, a scalar falls back to the non-enum any_of range."""
    tr = _make_mixed_any_of_transformer()
    result = tr.map_object({"id": "r1", "mixed": source_value}, source_type="Rec")
    assert result["mixed"] == source_value


@pytest.mark.parametrize("source_value", ["free text", "light_blue", "42"])
def test_enum_only_any_of_scalar_unmatched_is_none(source_value: str) -> None:
    """With no non-enum range to fall back to, an unmatched scalar becomes None."""
    tr = _make_mixed_any_of_transformer()
    result = tr.map_object({"id": "r1", "enums_only": source_value}, source_type="Rec")
    assert result["enums_only"] is None


def test_mixed_any_of_multivalued_maps_each_element_independently() -> None:
    """Elements of one multivalued slot resolve by different routes.

    ``light_red``/``dark_red`` map via TgtPrimary, ``dark_cyan`` via
    TgtSecondary, ``free text`` falls back to the non-enum ``string`` range,
    and a null element stays null rather than being stringified.
    """
    tr = _make_mixed_any_of_transformer()
    source = {"id": "r1", "mixed_multi": ["light_red", "dark_cyan", "free text", "dark_red", None]}
    result = tr.map_object(source, source_type="Rec")
    assert result["mixed_multi"] == ["red", "cyan", "free text", "red", None]


def test_enum_only_any_of_multivalued_unmatched_elements_are_none() -> None:
    """Without a non-enum range, only the enum-mapped elements survive; the rest are None."""
    tr = _make_mixed_any_of_transformer()
    source = {"id": "r1", "enums_only_multi": ["light_red", "dark_cyan", "free text", None]}
    result = tr.map_object(source, source_type="Rec")
    assert result["enums_only_multi"] == ["red", "cyan", None, None]


@pytest.mark.parametrize(
    ("source_value", "expected"),
    [
        ("light_red", "red"),
        (42, 42),
        (-1, -1),
    ],
)
def test_mixed_any_of_non_enum_range_need_not_be_string(source_value: object, expected: object) -> None:
    """The non-enum fallback is not string-specific; an integer range works the same way."""
    tr = _make_mixed_any_of_transformer()
    result = tr.map_object({"id": "r1", "mixed_int": source_value}, source_type="Rec")
    assert result["mixed_int"] == expected


# ---------------------------------------------------------------------------
# Nulls must survive target-range coercion
#
# Declaring ``range:`` on a slot derivation switches on two coercion steps that
# both used to stringify nulls: the cast in ObjectTransformer.map_object and
# Transformer._coerce_datatype. A scalar null is short-circuited by
# _derive_slot before either runs, but a null *element* of a multivalued slot
# reaches both, and "None"/"nan" strings (or a TypeError from int(None)) are
# never the right answer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("null_value", NULLISH)
def test_declared_range_does_not_coerce_null_scalar(null_value: float | None) -> None:
    """A null scalar stays null even when the slot derivation declares a range."""
    tr = _make_mixed_any_of_transformer(slot_ranges={"mixed": "string"})
    result = tr.map_object({"id": "r1", "mixed": null_value}, source_type="Rec")
    assert result["mixed"] is None


@pytest.mark.parametrize("null_value", NULLISH)
def test_declared_range_does_not_coerce_null_in_non_enum_fallback(null_value: float | None) -> None:
    """A null element routed through the non-enum any_of fallback stays null."""
    tr = _make_mixed_any_of_transformer(slot_ranges={"mixed_multi": "string"})
    source = {"id": "r1", "mixed_multi": ["light_red", null_value, "free text"]}
    result = tr.map_object(source, source_type="Rec")
    assert result["mixed_multi"] == ["red", None, "free text"]


@pytest.mark.parametrize("null_value", NULLISH)
def test_declared_range_does_not_coerce_null_in_plain_typed_slot(null_value: float | None) -> None:
    """The same holds for a plain typed multivalued slot, with no any_of involved."""
    tr = _make_mixed_any_of_transformer(slot_ranges={"plain_strings": "string"})
    result = tr.map_object({"id": "r1", "plain_strings": ["a", null_value, "b"]}, source_type="Rec")
    assert result["plain_strings"] == ["a", None, "b"]


@pytest.mark.parametrize("null_value", NULLISH)
def test_integer_range_null_element_does_not_raise(null_value: float | None) -> None:
    """``int(None)`` would raise TypeError; the null passes through untouched."""
    tr = _make_mixed_any_of_transformer(slot_ranges={"plain_ints": "integer"})
    result = tr.map_object({"id": "r1", "plain_ints": [1, null_value, 2]}, source_type="Rec")
    assert result["plain_ints"] == [1, None, 2]


@pytest.mark.parametrize("target_type", ["string", "integer", "float", "double", "uri", "curie", None])
@pytest.mark.parametrize("null_value", NULLISH)
def test_map_object_never_coerces_a_null_of_scalar_type(target_type: str | None, null_value: float | None) -> None:
    """map_object maps a null of scalar type to None for every target type."""
    tr = _make_mixed_any_of_transformer()
    assert tr.map_object(null_value, source_type="string", target_type=target_type) is None


@pytest.mark.parametrize("target_range", ["string", "integer", "float", "boolean"])
@pytest.mark.parametrize("null_value", NULLISH)
def test_coerce_datatype_never_coerces_a_null(target_range: str, null_value: float | None) -> None:
    """_coerce_datatype leaves nulls alone, standalone and nested in a list."""
    tr = _make_mixed_any_of_transformer()
    assert tr._coerce_datatype(null_value, target_range) is None
    assert tr._coerce_datatype([null_value], target_range) == [None]
    assert tr._coerce_datatype({"k": null_value}, target_range) == {"k": None}
