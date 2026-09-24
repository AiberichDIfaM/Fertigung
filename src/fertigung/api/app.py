import csv
import io
import math
import os
import shutil
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fertigung.api.store import Store, now
from fertigung.api.worker import TrainingWorker, db_path, model_dir
from fertigung.core.config import PlantConfig, reference_config
from fertigung.core.plant import Plant
from fertigung.core.validation import material_costs, validate
from fertigung.evaluation import compare, run_episode
from fertigung.heuristics import POLICIES, make_policy
from fertigung.rl.env import default_horizon
from fertigung.rl.model import META_FILE, MODEL_FILE, TrainedModel
from fertigung.rl.train import TrainingConfig

UI_DIR = Path(__file__).parent.parent / "ui"


class TrainingJobCreate(BaseModel):
    plant_id: str
    training: TrainingConfig = TrainingConfig()


class PlantSource(BaseModel):
    plant_id: str | None = Field(None, description="Stored plant; alternatively pass `config`")
    config: PlantConfig | None = None
    model_id: str | None = Field(None, description="Trained model; its plant is used if no plant is given")
    ticks: int | None = Field(None, ge=1, description="Default: model horizon or 1.2 x last deadline")


class SimulationCreate(PlantSource):
    policy: Literal["pull", "fifo", "random", "model"] = "pull"
    seed: int = 0


class EvaluationCreate(PlantSource):
    episodes: int = Field(5, ge=1, le=50, description="Episodes for the random baseline")


def analyze(config: PlantConfig) -> dict:
    plant = Plant(config)
    cost = material_costs(plant)
    return {
        "issues": [asdict(i) for i in validate(config)],
        "raw": plant.raw,
        "intermediate": plant.intermediate,
        "final": plant.final,
        "material_costs": {p: (None if math.isinf(c) else c) for p, c in cost.items()},
        "edges": [
            {"transformation": t.name, "input": p, "count": n, "output": t.output}
            for i, t in enumerate(plant.transformations)
            if i in plant.assigned_transformations
            for p, n in t.inputs.items()
        ],
    }


def gantt(events) -> list[dict]:
    bars = {}
    for e in events:
        if e.kind == "dispatch":
            bars[e.job] = {
                "job": e.job,
                "machine": e.machine,
                "transformation": e.transformation,
                "output": e.part_type,
                "start": e.time,
                "end": None,
                "blocked_from": None,
            }
        elif e.kind == "blocked":
            bars[e.job]["blocked_from"] = e.time
        else:
            bars[e.job]["end"] = e.time
    return list(bars.values())


def training_curve(path: Path) -> list[dict]:
    progress = path / "progress.csv"
    if not progress.exists():
        return []
    points = []
    with progress.open(newline="") as f:
        for row in csv.DictReader(f):
            point = {"timesteps": int(float(row["time/total_timesteps"]))}
            for key, column in (("train_reward", "rollout/ep_rew_mean"), ("eval_reward", "eval/mean_reward")):
                if row.get(column):
                    point[key] = float(row[column])
            if len(point) > 1:
                points.append(point)
    return points


def create_app(data_dir: str | Path | None = None, start_worker: bool = True) -> FastAPI:
    data = Path(data_dir or os.environ.get("FERTIGUNG_DATA_DIR", "data")).resolve()
    store = Store(db_path(data))
    worker = TrainingWorker(store, data)
    if not store.list("plants"):
        ref = reference_config()
        store.insert("plants", name=ref.name, config=ref.model_dump(), updated_at=now())

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="Fertigung", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    @lru_cache(maxsize=8)
    def load_model(model_id: str) -> TrainedModel:
        model = store.get("models", model_id)
        if model is None:
            raise HTTPException(404, f"model {model_id} not found")
        return TrainedModel(data / model["path"])

    def get_or_404(table: str, id: str) -> dict:
        row = store.get(table, id)
        if row is None:
            raise HTTPException(404, f"{table[:-1]} {id} not found")
        return row

    def resolve(source: PlantSource) -> tuple[PlantConfig, TrainedModel | None, int]:
        trained = load_model(source.model_id) if source.model_id else None
        if source.config is not None:
            config = source.config
        elif source.plant_id:
            config = PlantConfig.model_validate(get_or_404("plants", source.plant_id)["config"])
        elif trained:
            config = trained.config
        else:
            raise HTTPException(422, "pass plant_id, config or model_id")
        ticks = source.ticks or (trained.horizon if trained else default_horizon(config))
        return config, trained, ticks

    def model_policy(trained: TrainedModel, config: PlantConfig):
        try:
            return trained.policy(config)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/policies")
    def policies():
        return sorted(POLICIES) + ["model"]

    @app.get("/plants")
    def list_plants():
        return store.list("plants")

    @app.post("/plants", status_code=201)
    def create_plant(config: PlantConfig):
        return store.insert("plants", name=config.name, config=config.model_dump(), updated_at=now())

    @app.post("/plants/validate")
    def validate_config(config: PlantConfig):
        return analyze(config)

    @app.get("/plants/{plant_id}")
    def get_plant(plant_id: str):
        return get_or_404("plants", plant_id)

    @app.put("/plants/{plant_id}")
    def update_plant(plant_id: str, config: PlantConfig):
        get_or_404("plants", plant_id)
        return store.update(
            "plants", plant_id, name=config.name, config=config.model_dump(), updated_at=now()
        )

    @app.delete("/plants/{plant_id}", status_code=204)
    def delete_plant(plant_id: str):
        if not store.delete("plants", plant_id):
            raise HTTPException(404, f"plant {plant_id} not found")

    @app.post("/plants/{plant_id}/validate")
    def validate_plant(plant_id: str):
        return analyze(PlantConfig.model_validate(get_or_404("plants", plant_id)["config"]))

    @app.post("/training-jobs", status_code=201)
    def create_job(body: TrainingJobCreate):
        plant = get_or_404("plants", body.plant_id)
        if any(i["level"] == "error" for i in analyze(PlantConfig.model_validate(plant["config"]))["issues"]):
            raise HTTPException(422, "plant has validation errors")
        return store.insert(
            "jobs",
            plant_id=plant["id"],
            plant_name=plant["name"],
            plant=plant["config"],
            status="queued",
            training=body.training.model_dump(),
        )

    @app.get("/training-jobs")
    def list_jobs():
        return store.list("jobs")

    @app.get("/training-jobs/{job_id}")
    def get_job(job_id: str):
        job = get_or_404("jobs", job_id)
        return job | {"curve": training_curve(model_dir(data, job_id))}

    @app.delete("/training-jobs/{job_id}")
    def cancel_job(job_id: str):
        job = get_or_404("jobs", job_id)
        if job["status"] == "queued":
            return store.update("jobs", job_id, status="cancelled", finished_at=now())
        if job["status"] == "running":
            return store.update("jobs", job_id, status="cancelling")
        raise HTTPException(409, f"job is {job['status']}")

    @app.get("/models")
    def list_models():
        return store.list("models")

    @app.get("/models/{model_id}")
    def get_model(model_id: str):
        return get_or_404("models", model_id)

    @app.get("/models/{model_id}/download")
    def download_model(model_id: str):
        model = get_or_404("models", model_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in (MODEL_FILE, META_FILE, "progress.csv"):
                if (data / model["path"] / name).exists():
                    archive.write(data / model["path"] / name, name)
        headers = {"Content-Disposition": f'attachment; filename="{model["name"].replace(" ", "_")}.zip"'}
        return Response(buffer.getvalue(), media_type="application/zip", headers=headers)

    @app.delete("/models/{model_id}", status_code=204)
    def delete_model(model_id: str):
        model = get_or_404("models", model_id)
        store.delete("models", model_id)
        load_model.cache_clear()
        shutil.rmtree(data / model["path"], ignore_errors=True)

    @app.post("/simulations", status_code=201)
    def create_simulation(body: SimulationCreate):
        config, trained, ticks = resolve(body)
        if body.policy == "model":
            if trained is None:
                raise HTTPException(422, "policy 'model' needs model_id")
            policy = model_policy(trained, config)
        else:
            policy = make_policy(body.policy, body.seed)
        sim, reward = run_episode(config, policy, ticks)
        result = {
            "kpis": sim.kpis(),
            "reward": reward.total,
            "reward_components": dict(reward.totals),
            "orders": [asdict(o) | {"lateness": o.lateness(sim.time)} for o in sim.orders],
            "gantt": gantt(sim.events),
            "events": [asdict(e) for e in sim.events],
        }
        request = body.model_dump(exclude={"config"}) | {"plant": config.name, "ticks": ticks}
        return store.insert("simulations", request=request, result=result)

    @app.get("/simulations/{simulation_id}")
    def get_simulation(simulation_id: str):
        return get_or_404("simulations", simulation_id)

    @app.post("/evaluations")
    def create_evaluation(body: EvaluationCreate):
        config, trained, ticks = resolve(body)
        extra = {"model": lambda seed: model_policy(trained, config)} if trained else None
        return {"plant": config.name, "ticks": ticks, "results": compare(config, ticks, extra, body.episodes)}

    return app
