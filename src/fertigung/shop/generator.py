"""Random order books on the workshop layout, for benchmarks and training. `load` scales the number of
orders; at 1.0 the shop is busy but `pull` meets nearly all deadlines, from about 1.6 on orders get late."""

import random

from fertigung.shop.config import ShopConfig, workshop_config

# Pickups at 15:00: Tue-Fri of the first week, Mon and Tue of the second (minute 0 is Monday 06:00).
DEADLINES = [1980, 3420, 4860, 6300, 540 + 7 * 1440, 1980 + 7 * 1440]
QUANTITIES = {"frame": (3, 10), "housing": (2, 7), "bracket": (8, 25)}


def random_shop(seed: int, load: float = 1.0) -> ShopConfig:
    rng = random.Random(seed)
    data = workshop_config().model_dump(by_alias=True)
    orders = []
    for _ in range(int(rng.randint(12, 18) * load)):
        product = rng.choice(list(QUANTITIES))
        deadline = rng.choice(DEADLINES)
        release = max(0, deadline - rng.choice([1, 2, 3, 4]) * 1440)
        orders.append(
            {
                "product": product,
                "quantity": rng.randint(*QUANTITIES[product]),
                "release": 0 if release < 1440 else release - 540,  # 06:00 of that day
                "deadline": deadline,
                "priority": rng.choice([1, 1, 1, 2]),
            }
        )
    data["orders"] = sorted(orders, key=lambda o: o["deadline"])
    data["name"] = f"random-{seed}-{load:g}"
    return ShopConfig.model_validate(data)
