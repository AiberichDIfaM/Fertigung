import argparse
import json
import os
import sys

from fertigung.core.config import default_horizon, load_config, reference_config
from fertigung.core.validation import validate
from fertigung.evaluation import compare, format_table, run_episode
from fertigung.heuristics import POLICIES, make_policy


def _load(args):
    """Plant config, trained model (or None) and episode length from the CLI arguments."""
    trained = None
    if getattr(args, "model", None):
        from fertigung.rl.model import TrainedModel

        trained = TrainedModel(args.model)
    config = load_config(args.config) if args.config else trained.config if trained else reference_config()
    ticks = args.ticks or (trained.horizon_for(config) if trained else default_horizon(config))
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
        architecture="transfer" if args.transfer or args.init_model else "plant",
        generated_plants=args.generated_plants,
        init_model=args.init_model,
    )
    out = train(config, cfg, args.out)
    print(f"model written to {out}")
    return 0


def cmd_evaluate(args) -> int:
    config, trained, ticks = _load(args)
    extra = {"model": lambda seed: trained.policy(config)} if trained else None
    print(f"{config.name}, {ticks} ticks")
    print(format_table(compare(config, ticks, extra, args.episodes, args.policies)))
    return 0


def cmd_generate(args) -> int:
    import yaml

    from fertigung.generator import random_plant

    print(yaml.safe_dump(random_plant(args.seed).model_dump(exclude_defaults=True), sort_keys=False))
    return 0


def cmd_benchmark(args) -> int:
    from fertigung.evaluation import benchmark
    from fertigung.generator import random_plant
    from fertigung.heuristics import make_policy
    from fertigung.rl.model import TrainedModel

    trained = TrainedModel(args.model)
    plants = [random_plant(s) for s in range(args.seed, args.seed + args.plants)]
    factories = {
        "pull": lambda config, seed: make_policy("pull"),
        "fifo": lambda config, seed: make_policy("fifo"),
        "model": lambda config, seed: trained.policy(config),
    }
    results = benchmark(plants, factories, trained.horizon_for)
    print(f"{args.plants} generated plants, seeds {args.seed}..{args.seed + args.plants - 1}")
    print(format_table(results))
    for name, r in results.items():
        if name != "pull":
            print(
                f"{name}: better than pull on {r['better_than_pull']}, worse on {r['worse_than_pull']} plants"
            )
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
    p.add_argument("--transfer", action="store_true", help="plant-independent model that works on any plant")
    p.add_argument("--generated-plants", type=int, default=0, help="random plants to train on as well")
    p.add_argument("--init-model", help="transfer model to fine-tune, e.g. 'general'")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="compare heuristics and optionally a trained model")
    p.add_argument("config", nargs="?", help=config_help)
    p.add_argument("--model", help="trained model directory")
    p.add_argument("--ticks", type=int, help="episode length (default: model horizon or 1.2 x last deadline)")
    p.add_argument("--episodes", type=int, default=5, help="episodes for the random baseline")
    p.add_argument(
        "--policies", nargs="+", choices=sorted(POLICIES), help="heuristics to compare (default: all)"
    )
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("generate", help="print a random plant configuration as YAML")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("benchmark", help="compare a transfer model with the heuristics on generated plants")
    p.add_argument("--model", required=True, help="transfer model directory or bundled name")
    p.add_argument("--plants", type=int, default=20)
    p.add_argument(
        "--seed", type=int, default=50_000, help="seed of the first plant (keep clear of training seeds)"
    )
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", help="database and models (default: $FERTIGUNG_DATA_DIR or ./data)")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
