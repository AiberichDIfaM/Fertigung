import importlib.util
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from fertigung.core.config import PlantConfig
from fertigung.evaluation import evaluate
from fertigung.rl.env import JobShopEnv, default_horizon
from fertigung.rl.model import MODEL_FILE, TrainedModel, write_meta
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
    pretrain: Literal["pull", "fifo"] | None = Field("pull", description="Heuristic to imitate before PPO")
    pretrain_episodes: int = Field(30, ge=1)
    pretrain_epochs: int = Field(150, ge=1)


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

    env = make_vec_env(lambda: JobShopEnv(plant, horizon), n_envs=cfg.n_envs, seed=cfg.seed)
    eval_env = make_vec_env(lambda: JobShopEnv(plant, horizon), n_envs=1, seed=cfg.seed + 1000)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=plant.reward.gamma,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": cfg.net_arch},
        seed=cfg.seed,
        device="cpu",
    )
    formats = ["csv"] + (["tensorboard"] if importlib.util.find_spec("tensorboard") else [])
    model.set_logger(configure(str(out), formats))

    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes=1,
        best_model_save_path=str(out / "best"),
        log_path=str(out / "eval"),
        verbose=0,
    )
    if cfg.pretrain:
        data = expert_rollouts(
            plant, horizon, cfg.pretrain, cfg.pretrain_episodes, plant.reward.gamma, cfg.seed
        )
        behavior_cloning(model, data, cfg.pretrain_epochs)
        eval_callback.best_mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1)
        (out / "best").mkdir(exist_ok=True)
        model.save(out / "best" / "best_model.zip")

    callbacks = [
        eval_callback,
        CheckpointCallback(max(cfg.eval_freq * 5 // cfg.n_envs, 1), str(out / "checkpoints"), "model"),
    ]
    if callback is not None:
        callbacks.append(callback)
    model.learn(cfg.timesteps, callback=callbacks)
    model.logger.close()

    best = out / "best" / "best_model.zip"
    if best.exists():
        shutil.copyfile(best, out / MODEL_FILE)
    else:
        model.save(out / MODEL_FILE)

    meta = {"plant": plant.model_dump(), "horizon": horizon, "training": cfg.model_dump()}
    write_meta(out, **meta)
    trained = TrainedModel(out)
    write_meta(out, **meta, evaluation=evaluate(plant, lambda seed: trained.policy(), horizon))
    return out
