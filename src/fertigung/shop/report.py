"""Run a shop simulation and turn it into a JSON-friendly report (KPIs, orders, Gantt data, trips, events)."""

from dataclasses import asdict
from importlib import resources

import yaml

from fertigung.shop.config import ShopConfig, load_shop
from fertigung.shop.heuristics import ShopPull
from fertigung.shop.model import ShopModel
from fertigung.shop.simulation import ShopSimulation

POLICIES = {"pull": ShopPull}


def workshop_config() -> ShopConfig:
    text = resources.files("fertigung.configs").joinpath("workshop.yaml").read_text(encoding="utf-8")
    return load_shop(yaml.safe_load(text))


def default_minutes(config: ShopConfig) -> int:
    """Until one day after the last deadline, at least one week."""
    return max([o.deadline + 1440 for o in config.orders] + [7 * 1440])


def simulate(
    config: ShopConfig, policy: str = "pull", minutes: int | None = None, queue_limit: int = 2
) -> dict:
    sim = ShopSimulation(ShopModel(config))
    sim.run(POLICIES[policy](queue_limit), minutes or default_minutes(config))
    model = sim.model
    jobs = []
    for job in sim.jobs.values():
        tr = model.transformations[job.transformation]
        setup = job.started is not None and job.run_start is not None and job.run_start > job.started
        jobs.append(
            {
                "id": job.id,
                "machine": model.machines[job.machine].name,
                "transformation": tr.name,
                "output": tr.output,
                "family": tr.family,
                "state": job.state,
                "setup_start": job.started if setup else None,
                "run_start": job.run_start,
                "run_end": job.run_end,
                "finished": job.finished,
                "paused": job.paused,
            }
        )
    return {
        "minutes": sim.time,
        "start": config.start.model_dump(),
        "kpis": sim.kpis(),
        "orders": [
            asdict(o) | {"on_time": o.completed_at is not None and o.completed_at <= o.deadline}
            for o in sim.orders
        ],
        "machines": [{"name": mc.name, "type": mc.type, "slots": mc.slots} for mc in model.machines],
        "vehicles": [v.name for v in sim.vehicles],
        "jobs": jobs,
        "trips": [asdict(t) for t in sim.trips],
        "events": [asdict(e) for e in sim.events],
    }
