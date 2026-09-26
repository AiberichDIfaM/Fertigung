import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant
from fertigung.core.simulation import Simulation
from fertigung.rl.env import CandidateObserver, Observer, default_horizon

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
        self.architecture = self.meta.get("architecture", "plant")
        self.max_candidates = self.meta.get("max_candidates")
        self.model = MaskablePPO.load(self.path / MODEL_FILE, device="cpu")

    def horizon_for(self, config: PlantConfig) -> int:
        """Episode length the model expects on `config`: its training horizon for plant-specific models."""
        return self.horizon if self.architecture == "plant" else default_horizon(config)

    def policy(self, config: PlantConfig | None = None) -> "ModelPolicy":
        config = config or self.config
        observer = make_observer(self.architecture, config, self.horizon_for(config), self.max_candidates)
        try:
            return ModelPolicy(self.model, observer)
        except ValueError:
            raise ValueError(
                f"this model is specific to plant '{self.config.name}' and does not fit '{config.name}'; "
                "use a transferable model"
            ) from None


def make_observer(architecture: str, config: PlantConfig, horizon: int, max_candidates: int | None):
    plant = Plant(config)
    if architecture == "plant":
        return Observer(plant, horizon)
    observer = CandidateObserver(max_candidates)
    observer.bind(plant, horizon)
    return observer


class ModelPolicy:
    """Deterministic dispatch policy backed by a (trained or training) MaskablePPO model."""

    def __init__(self, model: MaskablePPO, observer):
        self.model, self.observer = model, observer
        if (observer.size, observer.n_actions) != (model.observation_space.shape[0], model.action_space.n):
            raise ValueError("observation/action spaces do not match the model")

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        action, _ = self.model.predict(
            self.observer.observe(sim), action_masks=self.observer.action_mask(sim), deterministic=True
        )
        return self.observer.to_dispatch(int(action), sim)


def write_meta(path: Path, **meta):
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
