from stable_baselines3.common.env_checker import check_env

from fertigung.core.config import reference_config
from fertigung.evaluation import evaluate
from fertigung.rl.env import JobShopEnv
from fertigung.rl.model import TrainedModel
from fertigung.rl.train import TrainingConfig, train


def test_env_passes_check_env():
    check_env(JobShopEnv(reference_config()))


def test_train_save_load_evaluate(tmp_path):
    config = reference_config()
    cfg = TrainingConfig(
        timesteps=256,
        n_envs=1,
        n_steps=128,
        batch_size=64,
        eval_freq=128,
        pretrain_episodes=2,
        pretrain_epochs=2,
    )
    trained = TrainedModel(train(config, cfg, tmp_path))
    expected = evaluate(config, lambda seed: trained.policy(), trained.horizon)
    assert trained.meta["evaluation"] == expected
