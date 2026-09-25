import argparse
import json
import os
import sys

from fertigung.core.config import load_config, reference_config
from fertigung.core.validation import validate
from fertigung.evaluation import compare, format_table, run_episode
from fertigung.heuristics import POLICIES, make_policy


def _default_horizon(config):
    from fertigung.rl.env import default_horizon

    return default_horizon(config)


def _load(args):
    """Plant config, trained model (or None) and episode length from the CLI arguments."""
    trained = None
    if getattr(args, "model", None):
        from fertigung.rl.model import TrainedModel

        trained = TrainedModel(args.model)
    config = load_config(args.config) if args.config else trained.config if trained else reference_config()
    ticks = args.ticks or (trained.horizon if trained else _default_horizon(config))
    return config, trained, ticks


def cmd_validate(args) -> int:
    issues = validate(load_config(args.config) if args.config else reference_config())
    for issue in issues:
        print(f"{issue.level}: {issue.message}")
    if not issues:
        print("ok")
    return 1 if any(i.level == "error" for i in issues) else 0


def cmd_simulate(args) -> int:
    config, trained, ticks = _load(args)
    policy = trained.policy(config) if trained else make_policy(args.policy, args.seed)
    sim, reward = run_episode(config, policy, ticks)
    if args.events:
        for e in sim.events:
            print(json.dumps(e.__dict__))
    result = sim.kpis() | {"reward": reward.total, "reward_components": dict(reward.totals)}
    print(json.dumps(result, indent=2))
    return 0


def cmd_train(args) -> int:
    from fertigung.rl.train import TrainingConfig, train

    config = load_config(args.config) if args.config else reference_config()
    cfg = TrainingConfig(
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        horizon=args.ticks,
        pretrain=None if args.no_pretrain else "pull",
    )
    out = train(config, cfg, args.out)
    print(f"model written to {out}")
    return 0


def cmd_evaluate(args) -> int:
    config, trained, ticks = _load(args)
    extra = {"model": lambda seed: trained.policy(config)} if trained else None
    print(f"{config.name}, {ticks} ticks")
    print(format_table(compare(config, ticks, extra, args.episodes)))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from fertigung.api.app import create_app

    if args.host not in ("127.0.0.1", "localhost") and not os.environ.get("FERTIGUNG_API_KEY"):
        print("warning: serving on a public interface without FERTIGUNG_API_KEY", file=sys.stderr)
    uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fertigung")
    sub = parser.add_subparsers(dest="command", required=True)
    config_help = "YAML/JSON plant config (default: the model's plant or the reference plant)"

    p = sub.add_parser("validate", help="check a plant configuration")
    p.add_argument("config", nargs="?", help=config_help)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("simulate", help="run one episode with a heuristic or a trained model")
    p.add_argument("config", nargs="?", help=config_help)
    p.add_argument("--policy", choices=sorted(POLICIES), default="pull")
    p.add_argument("--model", help="trained model directory (overrides --policy)")
    p.add_argument("--ticks", type=int, help="episode length (default: model horizon or 1.2 x last deadline)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--events", action="store_true", help="print the event log as JSON lines")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("train", help="train a MaskablePPO dispatcher")
    p.add_argument("config", nargs="?", help=config_help)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--ticks", type=int, help="episode length (default: 1.2 x last deadline)")
    p.add_argument("--out", help="output directory (default: models/<plant>-<timestamp>)")
    p.add_argument("--no-pretrain", action="store_true", help="skip imitation of the pull heuristic")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="compare heuristics and optionally a trained model")
    p.add_argument("config", nargs="?", help=config_help)
    p.add_argument("--model", help="trained model directory")
    p.add_argument("--ticks", type=int, help="episode length (default: model horizon or 1.2 x last deadline)")
    p.add_argument("--episodes", type=int, default=5, help="episodes for the random baseline")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", help="database and models (default: $FERTIGUNG_DATA_DIR or ./data)")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
