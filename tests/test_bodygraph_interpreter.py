from dataclasses import is_dataclass

import pytest

from human_design.vision.constants import (
    ALL_CHANNELS,
    CANONICAL_CENTERS,
    CHANNEL_TO_CENTERS,
)
from human_design.vision.interpreter import (
    BodyGraphInterpretationResult,
    derive_active_channels,
    derive_active_gates,
    derive_defined_centers,
    interpret_bodygraph,
)
from human_design.vision.models import (
    Activation,
    DerivedBasicInfo,
    DerivedChartData,
    DesignActivationColumn,
    PersonalityActivationColumn,
    RawVisionExtraction,
    ValidationCode,
    ValidationSeverity,
    ValidationSource,
)


PLANET_FIELDS = (
    "sun",
    "earth",
    "north_node",
    "south_node",
    "moon",
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
)

TYPE_FIELDS = {
    "Generator": ("To Respond", "Frustration", "Satisfaction"),
    "Manifesting Generator": ("To Respond", "Frustration", "Satisfaction"),
    "Projector": ("Wait for the Invitation", "Bitterness", "Success"),
    "Manifestor": ("To Inform", "Anger", "Peace"),
    "Reflector": ("Wait a Lunar Cycle", "Disappointment", "Surprise"),
}


def _activation(gate: int, line: int = 1) -> Activation:
    return Activation(gate=gate, line=line)


def _personality_column(
    values: dict[str, Activation | None] | None = None,
) -> PersonalityActivationColumn:
    column_values = {field_name: None for field_name in PLANET_FIELDS}
    column_values.update(values or {})
    return PersonalityActivationColumn(**column_values)


def _design_column(
    values: dict[str, Activation | None] | None = None,
) -> DesignActivationColumn:
    column_values = {field_name: None for field_name in PLANET_FIELDS}
    column_values.update(values or {})
    return DesignActivationColumn(**column_values)


def _raw_vision(
    *,
    personality_values: dict[str, Activation | None] | None = None,
    design_values: dict[str, Activation | None] | None = None,
    visually_defined_centers: tuple[str, ...] = (),
    visually_active_gates: tuple[int, ...] = (),
    visible_colored_channels: tuple[str, ...] = (),
) -> RawVisionExtraction:
    return RawVisionExtraction(
        personality=_personality_column(personality_values),
        design=_design_column(design_values),
        visually_defined_centers=visually_defined_centers,
        visually_active_gates=visually_active_gates,
        visible_colored_channels=visible_colored_channels,
    )


def _raw_with_gates(
    gates: tuple[int, ...],
    *,
    visually_defined_centers: tuple[str, ...] = (),
    visually_active_gates: tuple[int, ...] = (),
    visible_colored_channels: tuple[str, ...] = (),
) -> RawVisionExtraction:
    personality: dict[str, Activation | None] = {}
    design: dict[str, Activation | None] = {}
    positions = (
        *(
            ("personality", field_name)
            for field_name in PLANET_FIELDS
        ),
        *(("design", field_name) for field_name in PLANET_FIELDS),
    )

    for index, gate in enumerate(gates):
        column_name, field_name = positions[index]
        target = personality if column_name == "personality" else design
        target[field_name] = _activation(gate, line=(index % 6) + 1)

    return _raw_vision(
        personality_values=personality,
        design_values=design,
        visually_defined_centers=visually_defined_centers,
        visually_active_gates=visually_active_gates,
        visible_colored_channels=visible_colored_channels,
    )


def _interpret_gates(gates: tuple[int, ...]) -> BodyGraphInterpretationResult:
    return interpret_bodygraph(_raw_with_gates(gates))


def test_public_api_returns_typed_frozen_result_models() -> None:
    assert is_dataclass(BodyGraphInterpretationResult)
    assert BodyGraphInterpretationResult.__dataclass_params__.frozen is True

    result = interpret_bodygraph(_raw_with_gates((3, 60)))

    assert isinstance(result, BodyGraphInterpretationResult)
    assert isinstance(result.derived_chart_data, DerivedChartData)
    assert isinstance(result.derived_chart_data.basic_info, DerivedBasicInfo)
    assert isinstance(result.warnings, tuple)


def test_active_gates_derive_from_all_activation_fields_only() -> None:
    raw = _raw_vision(
        personality_values={
            "sun": _activation(60, 1),
            "earth": None,
            "north_node": _activation(3, 2),
            "south_node": _activation(3, 3),
            "moon": _activation(99, 1),
        },
        design_values={
            "sun": _activation(10, 4),
            "earth": _activation(57, 5),
            "north_node": _activation(0, 1),
            "moon": None,
        },
        visually_active_gates=(20, 34),
    )

    active_gates = derive_active_gates(raw)

    assert active_gates == (3, 10, 57, 60)
    assert 20 not in active_gates
    assert 34 not in active_gates
    assert 99 not in active_gates
    assert 0 not in active_gates


@pytest.mark.parametrize(
    ("gates", "expected_channel"),
    [
        ({3, 60}, "3-60"),
        ({10, 57}, "10-57"),
        ({34, 57}, "34-57"),
        ({10, 34}, "10-34"),
    ],
)
def test_active_channels_derive_required_canonical_pairs(
    gates: set[int],
    expected_channel: str,
) -> None:
    assert derive_active_channels(gates) == (expected_channel,)


def test_active_channels_use_all_channels_order_and_never_fake_channels() -> None:
    active_gates = {3, 10, 34, 57, 60, 99}
    expected = tuple(
        channel
        for channel in ALL_CHANNELS
        if all(int(gate) in active_gates for gate in channel.split("-"))
    )

    active_channels = derive_active_channels(active_gates)

    assert active_channels == expected
    assert active_channels == ("3-60", "10-34", "10-57", "34-57")
    assert "34-99" not in active_channels
    assert set(active_channels).issubset(ALL_CHANNELS)


def test_visible_colored_channels_do_not_become_derived_active_channels() -> None:
    raw = _raw_with_gates((1,), visible_colored_channels=("20-34", "34-57"))

    result = interpret_bodygraph(raw)

    assert result.derived_chart_data.active_gates == (1,)
    assert result.derived_chart_data.active_channels == ()


@pytest.mark.parametrize(
    ("active_channels", "expected_centers"),
    [
        ((), ()),
        (("3-60",), ("Sacral", "Root")),
        (
            ("3-60", "10-34", "30-41"),
            ("G", "Sacral", "Solar Plexus", "Root"),
        ),
    ],
)
def test_defined_centers_derive_from_active_channel_endpoints(
    active_channels: tuple[str, ...],
    expected_centers: tuple[str, ...],
) -> None:
    assert derive_defined_centers(active_channels) == expected_centers


def test_defined_centers_are_returned_in_canonical_center_order() -> None:
    assert derive_defined_centers(("24-61",)) == ("Head", "Ajna")


def test_raw_visual_centers_do_not_define_derived_centers_or_type() -> None:
    raw = _raw_with_gates(
        (1,),
        visually_defined_centers=("Sacral", "Throat"),
    )

    result = interpret_bodygraph(raw)

    assert result.derived_chart_data.active_channels == ()
    assert result.derived_chart_data.defined_centers == ()
    assert result.derived_chart_data.basic_info.type == "Reflector"


@pytest.mark.parametrize(
    ("channel", "centers"),
    [
        ("3-60", ("Sacral", "Root")),
        ("10-34", ("G", "Sacral")),
        ("10-57", ("G", "Spleen")),
        ("20-34", ("Throat", "Sacral")),
        ("21-45", ("Ego", "Throat")),
        ("24-61", ("Ajna", "Head")),
        ("37-40", ("Solar Plexus", "Ego")),
    ],
)
def test_representative_channel_mappings_are_consumed(
    channel: str,
    centers: tuple[str, str],
) -> None:
    derived_centers = derive_defined_centers((channel,))

    assert CHANNEL_TO_CENTERS[channel] == centers
    assert set(derived_centers) == set(centers)
    assert derived_centers == tuple(
        center for center in CANONICAL_CENTERS if center in centers
    )


def test_profile_derives_from_personality_and_design_sun_lines() -> None:
    raw = _raw_vision(
        personality_values={"sun": _activation(61, 4)},
        design_values={"sun": _activation(32, 6)},
    )

    result = interpret_bodygraph(raw)

    assert result.derived_chart_data.basic_info.profile == "4/6"


@pytest.mark.parametrize(
    ("personality_sun", "design_sun"),
    [
        (None, _activation(32, 6)),
        (_activation(61, 4), None),
    ],
)
def test_missing_sun_returns_unknown_profile_without_interpreter_warnings(
    personality_sun: Activation | None,
    design_sun: Activation | None,
) -> None:
    raw = _raw_vision(
        personality_values={"sun": personality_sun},
        design_values={"sun": design_sun},
    )

    result = interpret_bodygraph(raw)

    assert result.derived_chart_data.basic_info.profile == "Unknown"
    assert result.warnings == ()


@pytest.mark.parametrize(
    (
        "gates",
        "expected_channels",
        "expected_centers",
        "expected_type",
        "expected_authority",
    ),
    [
        (
            (3, 60),
            ("3-60",),
            ("Sacral", "Root"),
            "Generator",
            "Sacral",
        ),
        (
            (20, 34),
            ("20-34",),
            ("Throat", "Sacral"),
            "Manifesting Generator",
            "Sacral",
        ),
        (
            (10, 57),
            ("10-57",),
            ("G", "Spleen"),
            "Projector",
            "Splenic",
        ),
        (
            (21, 45),
            ("21-45",),
            ("Throat", "Ego"),
            "Manifestor",
            "Ego",
        ),
        (
            (1, 2),
            (),
            (),
            "Reflector",
            "Lunar",
        ),
    ],
)
def test_interpret_bodygraph_derives_type_authority_and_type_fields(
    gates: tuple[int, ...],
    expected_channels: tuple[str, ...],
    expected_centers: tuple[str, ...],
    expected_type: str,
    expected_authority: str,
) -> None:
    result = _interpret_gates(gates)
    chart = result.derived_chart_data
    basic_info = chart.basic_info
    strategy, not_self_theme, signature = TYPE_FIELDS[expected_type]

    assert chart.active_channels == expected_channels
    assert chart.defined_centers == expected_centers
    assert basic_info.type == expected_type
    assert basic_info.authority == expected_authority
    assert basic_info.strategy == strategy
    assert basic_info.not_self_theme == not_self_theme
    assert basic_info.signature == signature
    assert result.warnings == ()


def test_projector_with_25_51_has_ego_projected_authority() -> None:
    result = _interpret_gates((25, 51))
    chart = result.derived_chart_data

    assert chart.active_channels == ("25-51",)
    assert chart.defined_centers == ("G", "Ego")
    assert chart.basic_info.type == "Projector"
    assert chart.basic_info.authority == "Ego-Projected"
    assert result.warnings == ()


@pytest.mark.parametrize("channel", ["1-8", "7-31", "10-20", "13-33"])
def test_projector_direct_g_to_throat_channels_have_self_projected_authority(
    channel: str,
) -> None:
    gates = tuple(int(gate) for gate in channel.split("-"))

    result = _interpret_gates(gates)
    chart = result.derived_chart_data

    assert chart.active_channels == (channel,)
    assert chart.defined_centers == ("Throat", "G")
    assert chart.basic_info.type == "Projector"
    assert chart.basic_info.authority == "Self-Projected"
    assert result.warnings == ()


def test_projector_splenic_authority_precedes_projected_authorities() -> None:
    result = _interpret_gates((10, 57, 25, 51))
    chart = result.derived_chart_data

    assert chart.active_channels == ("10-57", "25-51")
    assert chart.defined_centers == ("G", "Ego", "Spleen")
    assert chart.basic_info.type == "Projector"
    assert chart.basic_info.authority == "Splenic"
    assert result.warnings == ()


def test_multi_hop_motor_to_throat_path_does_not_qualify_as_direct() -> None:
    result = _interpret_gates((2, 14, 1, 8))
    chart = result.derived_chart_data

    assert chart.active_channels == ("1-8", "2-14")
    assert chart.defined_centers == ("Throat", "G", "Sacral")
    assert chart.basic_info.type == "Generator"
    assert chart.basic_info.type != "Manifesting Generator"


def test_definition_uses_connected_components_with_coarse_split_taxonomy() -> None:
    no_definition = _interpret_gates((1, 2))
    split = _interpret_gates((3, 60, 10, 57))
    single = _interpret_gates((3, 60, 10, 34, 30, 41))

    assert no_definition.derived_chart_data.basic_info.definition == "No Definition"

    split_definition = split.derived_chart_data.basic_info.definition
    assert split.derived_chart_data.active_channels == ("3-60", "10-57")
    assert split_definition == "Split Definition"
    assert split_definition not in {
        "Simple-Split Definition",
        "Wide-Split Definition",
    }

    assert single.derived_chart_data.active_channels == (
        "3-60",
        "10-34",
        "30-41",
    )
    assert single.derived_chart_data.defined_centers == (
        "G",
        "Sacral",
        "Solar Plexus",
        "Root",
    )
    assert single.derived_chart_data.basic_info.definition == "Single Definition"


def test_unsupported_authority_returns_needs_review_warning() -> None:
    result = _interpret_gates((24, 61))
    basic_info = result.derived_chart_data.basic_info

    assert result.derived_chart_data.active_channels == ("24-61",)
    assert result.derived_chart_data.defined_centers == ("Head", "Ajna")
    assert basic_info.type == "Projector"
    assert basic_info.authority == "Needs Review"
    assert len(result.warnings) == 1

    warning = result.warnings[0]
    assert warning.code is ValidationCode.UNSUPPORTED_AUTHORITY
    assert warning.severity is ValidationSeverity.WARNING
    assert warning.affects_validity is False
    assert warning.source is ValidationSource.interpreter
