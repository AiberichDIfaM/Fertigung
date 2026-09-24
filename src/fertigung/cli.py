import argparse
import json
import sys

from fertigung.core.config import load_config, reference_config
from fertigung.core.plant import Plant
from fertigung.core.reward import Reward
from fertigung.core.simulation import Simulation
from fertigung.core.validation import validate
from fertigung.heuristics import POLICIES


def _config(path):
    return load_config(path) if path else reference_config()


def cmd_validate(args) -> int:
    issues = validate(_config(args.config))
    for issue in issues:
        print(f"{issue.level}: {issue.message}")
    if not issues:
        print("ok")
    return 1 if any(i.level == "error" for i in issues) else 0


def cmd_simulate(args) -> int:
    sim = Simulation(Plant(_config(args.config)))
    reward = Reward(sim)
    sim.run(POLICIES[args.policy], args.ticks, reward)
    if args.events:
        for e in sim.events:
            print(json.dumps(e.__dict__))
    result = sim.kpis() | {"reward": reward.total, "reward_components": dict(reward.totals)}
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fertigung")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check a plant configuration")
    p.add_argument("config", nargs="?", help="YAML/JSON plant config (default: reference plant)")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("simulate", help="run a plant with a heuristic policy")
    p.add_argument("config", nargs="?", help="YAML/JSON plant config (default: reference plant)")
    p.add_argument("--policy", choices=sorted(POLICIES), default="pull")
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--events", action="store_true", help="print the event log as JSON lines")
    p.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
