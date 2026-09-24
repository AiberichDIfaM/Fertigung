import multiprocessing
import shutil
import threading
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from fertigung.api.store import Store, now

ACTIVE = ("running", "cancelling")


def db_path(data_dir: Path) -> Path:
    return data_dir / "fertigung.db"


def model_dir(data_dir: Path, job_id: str) -> Path:
    return data_dir / "models" / job_id


class ProgressCallback(BaseCallback):
    """Reports progress to the store and stops training when the job is being cancelled."""

    def __init__(self, store: Store, job_id: str, every: int = 2048):
        super().__init__()
        self.store, self.job_id, self.every, self._last = store, job_id, every, 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        job = self.store.update("jobs", self.job_id, timesteps_done=self.num_timesteps)
        return job["status"] != "cancelling"


def run_training(job_id: str, data_dir: str):
    from fertigung.core.config import PlantConfig
    from fertigung.rl.model import TrainedModel
    from fertigung.rl.train import TrainingConfig, train

    data = Path(data_dir)
    store = Store(db_path(data))
    job = store.get("jobs", job_id)
    out = model_dir(data, job_id)
    try:
        plant = PlantConfig.model_validate(job["plant"])
        train(plant, TrainingConfig.model_validate(job["training"]), out, ProgressCallback(store, job_id))
        if store.get("jobs", job_id)["status"] == "cancelling":
            shutil.rmtree(out, ignore_errors=True)
            store.update("jobs", job_id, status="cancelled", finished_at=now())
            return
        trained = TrainedModel(out)
        model = store.insert(
            "models",
            name=f"{plant.name} {now()[:16].replace('T', ' ')}",
            plant_name=plant.name,
            job_id=job_id,
            path=str(out.relative_to(data)),
            horizon=trained.horizon,
            evaluation=trained.meta["evaluation"],
        )
        store.update(
            "jobs",
            job_id,
            status="completed",
            model_id=model["id"],
            timesteps_done=job["training"]["timesteps"],
            finished_at=now(),
        )
    except Exception as e:
        shutil.rmtree(out, ignore_errors=True)
        store.update("jobs", job_id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=now())


class TrainingWorker:
    """Runs queued training jobs one at a time, each in a spawned process."""

    def __init__(self, store: Store, data_dir: Path, poll: float = 1.0):
        self.store, self.data_dir, self.poll = store, data_dir, poll
        self.process = None
        self.job_id = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        for job in self.store.list("jobs"):
            if job["status"] in ACTIVE:
                self.store.update("jobs", job["id"], status="failed", error="interrupted", finished_at=now())
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.store.update(
                "jobs", self.job_id, status="failed", error="server shutdown", finished_at=now()
            )

    def _loop(self):
        while not self._stop.wait(self.poll):
            self._reap()
            if self.process is None:
                self._launch_next()

    def _reap(self):
        if self.process is None or self.process.is_alive():
            return
        self.process.join()
        job = self.store.get("jobs", self.job_id)
        if job and job["status"] in ACTIVE:
            status = "cancelled" if job["status"] == "cancelling" else "failed"
            self.store.update(
                "jobs", self.job_id, status=status, error="training process exited", finished_at=now()
            )
        self.process = self.job_id = None

    def _launch_next(self):
        queued = self.store.list("jobs", status="queued")
        if not queued:
            return
        job = queued[-1]
        self.store.update("jobs", job["id"], status="running", started_at=now())
        ctx = multiprocessing.get_context("spawn")
        self.job_id = job["id"]
        self.process = ctx.Process(target=run_training, args=(job["id"], str(self.data_dir)), daemon=True)
        self.process.start()
