import importlib.util
import math
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from fertigung.core.config import PlantConfig
from fertigung.evaluation import evaluate
from fertigung.generator import random_plant
from fertigung.rl.env import JobShopEnv, default_horizon
from fertigung.rl.model import MODEL_FILE, ModelPolicy, TrainedModel, make_observer, model_path, write_meta
from fertigung.rl.policy import CandidatePolicy
from fertigung.rl.pretrain import behavior_cloning, expert_rollouts


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timesteps: int = Field(500_000, ge=1)
    seed: int = 0
    horizon: int | None = Field(
        None, ge=1, description="Episode length in ticks; default 1.2 x last deadline"
    )
    n_envs: int = Field(4, ge=1)
    learning_rate: float = Field(1e-4, gt=0)
    n_steps: int = Field(1024, ge=8)
    batch_size: int = Field(256, ge=8)
    n_epochs: int = Field(10, ge=1)
    ent_coef: float = Field(0.001, ge=0)
    net_arch: list[int] = [256, 256]
    eval_freq: int = Field(20_000, ge=1, description="Timesteps between evaluations")
    pretrain: Literal["pull", "lookahead", "fifo"] | None = Field(
        "pull", description="Heuristic to imitate before PPO (lookahead: better but slow teacher)"
    )
    pretrain_episodes: int = Field(30, ge=1)
    pretrain_workers: int = Field(1, ge=1, description="Processes generating imitation data")
    pretrain_epochs: int = Field(150, ge=1)
    architecture: Literal["plant", "transfer"] = Field(
        "plant",
        description="plant: fixed to this plant; transfer: scores dispatch candidates, works on any plant",
    )
    generated_plants: int = Field(0, ge=0, description="transfer: random plants trained on besides this one")
    eval_plants: int = Field(
        4, ge=0, description="transfer: extra random plants evaluated on (with generated_plants)"
    )
    generator_seed: int = Field(10_000, description="transfer: seed of the first generated plant")
    max_candidates: int = Field(64, ge=1, description="transfer: dispatch candidates scored per decision")
    init_model: str | None = Field(None, description="transfer: model directory or bundled name to fine-tune")


class SelectionCallback(BaseCallback):
    """Every `every` steps, scores the policy with `score` and saves it to `path` if it is the best so far."""

    def __init__(self, score, path: Path, every: int):
        super().__init__()
        self.score, self.path, self.every = score, path, every
        self.best, self._last = -math.inf, 0

    def check(self, model) -> float:
        value = self.score(model)
        if value > self.best:
            self.best = value
            model.save(self.path)
        return value

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.logger.record("eval/mean_reward", self.check(self.model))
        return True


def default_output_dir(plant: PlantConfig) -> Path:
    return Path("models") / f"{plant.name}-{datetime.now():%Y%m%d-%H%M%S}"


def train(
    plant: PlantConfig,
    cfg: TrainingConfig | None = None,
    out_dir: str | Path | None = None,
    callback: BaseCallback | None = None,
) -> Path:
    """Train MaskablePPO; writes model.zip (best evaluated policy), meta.json and progress.csv."""
    cfg = cfg or TrainingConfig()
    out = Path(out_dir) if out_dir else default_output_dir(plant)
    out.mkdir(parents=True, exist_ok=True)
    horizon = cfg.horizon or default_horizon(plant)
    transfer = cfg.architecture == "transfer"
    if transfer:
        # Extra evaluation plants only make sense when training on generated plants, not when fine-tuning.
        n_eval = cfg.eval_plants if cfg.generated_plants else 0
        seeds = iter(range(cfg.generator_seed, cfg.generator_seed + cfg.generated_plants + n_eval))
        train_plants = [plant] + [random_plant(next(seeds)) for _ in range(cfg.generated_plants)]
        eval_plants = [plant] + [random_plant(next(seeds)) for _ in range(n_eval)]
        env_kwargs = {
            "horizon": cfg.horizon,
            "architecture": "transfer",
            "max_candidates": cfg.max_candidates,
        }
    else:
        train_plants, eval_plants = plant, plant
        env_kwargs = {"horizon": horizon}

    env = make_vec_env(lambda: JobShopEnv(train_plants, **env_kwargs), n_envs=cfg.n_envs, seed=cfg.seed)
    model = MaskablePPO(
        CandidatePolicy if transfer else "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=plant.reward.gamma,
        ent_coef=cfg.ent_coef,
        policy_kwargs=(
            {"max_candidates": cfg.max_candidates, "hidden": cfg.net_arch[0]}
            if transfer
            else {"net_arch": cfg.net_arch}
        ),
        seed=cfg.seed,
        device="cpu",
    )
    formats = ["csv"] + (["tensorboard"] if importlib.util.find_spec("tensorboard") else [])
    model.set_logger(configure(str(out), formats))

    def horizon_for(config: PlantConfig) -> int:
        return (cfg.horizon or default_horizon(config)) if transfer else horizon

    def score(current) -> float:
        """Mean unshaped reward on the evaluation plants: the metric that is reported, not the shaped one."""
        return mean(
            evaluate(
                c,
                lambda seed, c=c: ModelPolicy(
                    current, make_observer(cfg.architecture, c, horizon_for(c), cfg.max_candidates)
                ),
                horizon_for(c),
            )["reward_unshaped"]
            for c in (eval_plants if transfer else [plant])
        )

    selection = SelectionCallback(score, out / "best_model.zip", cfg.eval_freq)
    if cfg.init_model:
        model.set_parameters(str(model_path(cfg.init_model) / MODEL_FILE), device="cpu")
    elif cfg.pretrain:
        data = expert_rollouts(
            train_plants if transfer else [plant],
            env_kwargs,
            cfg.pretrain,
            cfg.pretrain_episodes,
            plant.reward.gamma,
            cfg.seed,
            cfg.pretrain_workers,
        )
        behavior_cloning(model, data, cfg.pretrain_epochs)
    if cfg.init_model or cfg.pretrain:
        model.logger.record("eval/mean_reward", selection.check(model))

    callbacks = [
        selection,
        CheckpointCallback(max(cfg.eval_freq * 5 // cfg.n_envs, 1), str(out / "checkpoints"), "model"),
    ]
    if callback is not None:
        callbacks.append(callback)
    model.learn(cfg.timesteps, callback=callbacks)
    model.logger.close()

    best = out / "best_model.zip"
    if best.exists():
        shutil.copyfile(best, out / MODEL_FILE)
    else:
        model.save(out / MODEL_FILE)

    meta = {
        "plant": plant.model_dump(),
        "horizon": horizon,
        "architecture": cfg.architecture,
        "max_candidates": cfg.max_candidates if transfer else None,
        "training": cfg.model_dump(),
    }
    write_meta(out, **meta)
    trained = TrainedModel(out)
    write_meta(out, **meta, evaluation=evaluate(plant, lambda seed: trained.policy(), horizon))
    return out
