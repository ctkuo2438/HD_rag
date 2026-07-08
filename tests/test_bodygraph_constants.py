from human_design.vision.constants import (
    ALL_CHANNELS,
    CANONICAL_CENTERS,
    CENTER_ALIASES,
    CHANNEL_TO_CENTERS,
    MOTOR_CENTERS,
)


EXPECTED_CANONICAL_CENTERS = (
    "Head",
    "Ajna",
    "Throat",
    "G",
    "Ego",
    "Sacral",
    "Spleen",
    "Solar Plexus",
    "Root",
)

EXPECTED_CENTER_ALIASES = {
    "Heart": "Ego",
    "Will": "Ego",
    "Ego": "Ego",
    "G Center": "G",
    "Self": "G",
    "Identity": "G",
    "Emotional": "Solar Plexus",
    "Solar Plexus": "Solar Plexus",
}

EXPECTED_ALL_CHANNELS = (
    "1-8",
    "2-14",
    "3-60",
    "4-63",
    "5-15",
    "6-59",
    "7-31",
    "9-52",
    "10-20",
    "10-34",
    "10-57",
    "11-56",
    "12-22",
    "13-33",
    "16-48",
    "17-62",
    "18-58",
    "19-49",
    "20-34",
    "20-57",
    "21-45",
    "23-43",
    "24-61",
    "25-51",
    "26-44",
    "27-50",
    "28-38",
    "29-46",
    "30-41",
    "32-54",
    "34-57",
    "35-36",
    "37-40",
    "39-55",
    "42-53",
    "47-64",
)

EXPECTED_CHANNEL_TO_CENTERS = {
    "1-8": ("G", "Throat"),
    "2-14": ("G", "Sacral"),
    "3-60": ("Sacral", "Root"),
    "4-63": ("Ajna", "Head"),
    "5-15": ("Sacral", "G"),
    "6-59": ("Solar Plexus", "Sacral"),
    "7-31": ("G", "Throat"),
    "9-52": ("Sacral", "Root"),
    "10-20": ("G", "Throat"),
    "10-34": ("G", "Sacral"),
    "10-57": ("G", "Spleen"),
    "11-56": ("Ajna", "Throat"),
    "12-22": ("Throat", "Solar Plexus"),
    "13-33": ("G", "Throat"),
    "16-48": ("Throat", "Spleen"),
    "17-62": ("Ajna", "Throat"),
    "18-58": ("Spleen", "Root"),
    "19-49": ("Root", "Solar Plexus"),
    "20-34": ("Throat", "Sacral"),
    "20-57": ("Throat", "Spleen"),
    "21-45": ("Ego", "Throat"),
    "23-43": ("Throat", "Ajna"),
    "24-61": ("Ajna", "Head"),
    "25-51": ("G", "Ego"),
    "26-44": ("Ego", "Spleen"),
    "27-50": ("Sacral", "Spleen"),
    "28-38": ("Spleen", "Root"),
    "29-46": ("Sacral", "G"),
    "30-41": ("Solar Plexus", "Root"),
    "32-54": ("Spleen", "Root"),
    "34-57": ("Sacral", "Spleen"),
    "35-36": ("Throat", "Solar Plexus"),
    "37-40": ("Solar Plexus", "Ego"),
    "39-55": ("Root", "Solar Plexus"),
    "42-53": ("Sacral", "Root"),
    "47-64": ("Ajna", "Head"),
}


def test_canonical_centers_are_exact_ordered_nine_center_tuple() -> None:
    assert CANONICAL_CENTERS == EXPECTED_CANONICAL_CENTERS
    assert len(CANONICAL_CENTERS) == 9
    assert len(set(CANONICAL_CENTERS)) == len(CANONICAL_CENTERS)


def test_center_aliases_are_exact_and_target_canonical_centers() -> None:
    assert CENTER_ALIASES == EXPECTED_CENTER_ALIASES
    assert CENTER_ALIASES["Heart"] == "Ego"
    assert set(CENTER_ALIASES.values()).issubset(CANONICAL_CENTERS)


def test_all_channels_are_exact_ordered_thirty_six_channel_tuple() -> None:
    assert ALL_CHANNELS == EXPECTED_ALL_CHANNELS
    assert len(ALL_CHANNELS) == 36
    assert len(set(ALL_CHANNELS)) == len(ALL_CHANNELS)
    assert "57-10" not in ALL_CHANNELS
    assert "34-10" not in ALL_CHANNELS
    assert "60-3" not in ALL_CHANNELS


def test_channel_to_centers_exactly_matches_normative_mapping() -> None:
    assert set(CHANNEL_TO_CENTERS) == set(ALL_CHANNELS)
    assert CHANNEL_TO_CENTERS == EXPECTED_CHANNEL_TO_CENTERS


def test_each_channel_maps_to_two_distinct_canonical_centers() -> None:
    canonical_centers = set(CANONICAL_CENTERS)

    for centers in CHANNEL_TO_CENTERS.values():
        assert isinstance(centers, tuple)
        assert len(centers) == 2
        assert centers[0] != centers[1]
        assert set(centers).issubset(canonical_centers)


def test_representative_channel_center_mappings_are_explicit() -> None:
    assert CHANNEL_TO_CENTERS["3-60"] == ("Sacral", "Root")
    assert CHANNEL_TO_CENTERS["10-34"] == ("G", "Sacral")
    assert CHANNEL_TO_CENTERS["10-57"] == ("G", "Spleen")
    assert CHANNEL_TO_CENTERS["20-34"] == ("Throat", "Sacral")
    assert CHANNEL_TO_CENTERS["21-45"] == ("Ego", "Throat")
    assert CHANNEL_TO_CENTERS["24-61"] == ("Ajna", "Head")
    assert CHANNEL_TO_CENTERS["37-40"] == ("Solar Plexus", "Ego")


def test_motor_centers_are_exact_canonical_frozenset() -> None:
    assert isinstance(MOTOR_CENTERS, frozenset)
    assert MOTOR_CENTERS == frozenset({"Root", "Sacral", "Solar Plexus", "Ego"})
    assert MOTOR_CENTERS.issubset(CANONICAL_CENTERS)
