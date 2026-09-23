from fertigung.core.reference_plant import anlage


def test_reference_plant_builds():
    assert len(anlage.machines) == 10
    assert len(anlage.all_part_types) == 22
