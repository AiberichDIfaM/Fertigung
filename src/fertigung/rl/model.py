import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant
from fertigung.core.simulation import Simulation
from fertigung.rl.env import Observer

MODEL_FILE = "model.zip"
META_FILE = "meta.json"
BUNDLED_DIR = Path(__file__).parent.parent / "models"


def bundled_model_path(name: str = "reference") -> Path:
    return BUNDLED_DIR / name


def model_path(name_or_path: str | Path) -> Path:
    """A model directory, or the name of a model bundled with the package (e.g. "reference")."""
    path = Path(name_or_path)
    return path if path.exists() else bundled_model_path(str(name_or_path))


class TrainedModel:
    def __init__(self, path: str | Path):
        self.path = model_path(path)
        self.meta = json.loads((self.path / META_FILE).read_text(encoding="utf-8"))
        self.config = PlantConfig.model_validate(self.meta["plant"])
        self.horizon = self.meta["horizon"]
        self.model = MaskablePPO.load(self.path / MODEL_FILE, device="cpu")

    def policy(self, config: PlantConfig | None = None) -> "ModelPolicy":
        return ModelPolicy(self, config or self.config)


class ModelPolicy:
    """Deterministic dispatch policy backed by a trained model."""

    def __init__(self, trained: TrainedModel, config: PlantConfig):
        self.model = trained.model
        self.observer = Observer(Plant(config), trained.horizon)
        expected = (self.model.observation_space.shape[0], self.model.action_space.n)
        if (self.observer.size, self.observer.n_actions) != expected:
            raise ValueError(f"plant '{config.name}' does not match the model's observation/action spaces")

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        action, _ = self.model.predict(
            self.observer.observe(sim), action_masks=self.observer.action_mask(sim), deterministic=True
        )
        return self.observer.to_dispatch(int(action))


def write_meta(path: Path, **meta):
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
