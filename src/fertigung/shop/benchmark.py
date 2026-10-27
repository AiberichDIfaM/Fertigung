"""Compare solvers on random order books. Solver specs are names with optional parameters, e.g. `pull`,
`pull:queue_limit=1`, `auto:interval=720`; the first spec is the baseline for better/worse counts."""

import time
from concurrent.futures import ProcessPoolExecutor

from fertigung.shop.generator import random_shop
from fertigung.shop.model import ShopModel
from fertigung.shop.simulation import ShopSimulation
from fertigung.shop.solvers import make_solver

MINUTES = 10 * 1440


def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, params = spec.partition(":")
    kwargs = {}
    for item in filter(None, params.split(",")):
        key, value = item.split("=")
        kwargs[key] = float(value) if "." in value else int(value)
    return name, kwargs


def run_case(args: tuple[str, int, float]) -> dict:
    spec, seed, load = args
    name, kwargs = parse_spec(spec)
    sim = ShopSimulation(ShopModel(random_shop(seed, load)), record=False)
    start = time.perf_counter()
    sim.run(make_solver(name, **kwargs), MINUTES)
    kpis = sim.kpis()
    return {
        "objective": kpis["objective"],
        "on_time": kpis["orders_on_time"],
        "orders": len(sim.orders),
        "late_unit_hours": kpis["late_unit_hours"],
        "setup_hours": kpis["setup_hours"],
        "seconds": time.perf_counter() - start,
    }


def benchmark(specs: list[str], seeds: range, loads: tuple[float, ...], processes: int | None = None) -> dict:
    cases = [(seed, load) for load in loads for seed in seeds]
    with ProcessPoolExecutor(processes) as pool:
        results = {
            spec: list(pool.map(run_case, [(spec, seed, load) for seed, load in cases])) for spec in specs
        }
    return {"cases": cases, "results": results}


def format_table(report: dict) -> str:
    results = report["results"]
    base = next(iter(results.values()))
    lines = [
        f"{'solver':28} {'objective':>11} {'on time':>11} {'late h':>8} {'setup h':>8} {'better':>6} "
        f"{'worse':>5} {'s/run':>6}"
    ]
    for spec, rows in results.items():
        better = sum(r["objective"] > b["objective"] + 1 for r, b in zip(rows, base, strict=True))
        worse = sum(r["objective"] < b["objective"] - 1 for r, b in zip(rows, base, strict=True))
        on_time = f"{sum(r['on_time'] for r in rows)}/{sum(r['orders'] for r in rows)}"
        lines.append(
            f"{spec:28} {sum(r['objective'] for r in rows):11.0f} {on_time:>11} "
            f"{sum(r['late_unit_hours'] for r in rows):8.0f} {sum(r['setup_hours'] for r in rows):8.0f} "
            f"{better:6} {worse:5} {sum(r['seconds'] for r in rows) / len(rows):6.1f}"
        )
    return "\n".join(lines)
