from fertigung.core.validation import validate
from fertigung.generator import random_plant


def test_generated_plants_are_valid_and_reproducible():
    assert all(validate(random_plant(seed)) == [] for seed in range(10))
    assert random_plant(3) == random_plant(3)
