import time

from fastapi.testclient import TestClient

from fertigung.api.app import create_app
from fertigung.core.config import reference_config


def test_plants_simulations_evaluations(tmp_path):
    client = TestClient(create_app(tmp_path, start_worker=False))
    [plant] = client.get("/plants").json()
    assert plant["name"] == "reference"
    assert "x-data" in client.get("/").text

    config = reference_config().model_dump()
    config["machine_types"][0]["transformations"].remove("tr10")
    analysis = client.post("/plants/validate", json=config).json()
    assert any("tr10" in i["message"] for i in analysis["issues"])
    assert client.post("/plants", json={**config, "machines": []}).status_code == 422

    sim = client.post("/simulations", json={"plant_id": plant["id"]}).json()
    assert sim["result"]["kpis"]["orders_on_time"] == 4
    assert all(bar["end"] is None or bar["end"] > bar["start"] for bar in sim["result"]["gantt"])
    assert client.get(f"/simulations/{sim['id']}").json() == sim

    evaluation = client.post("/evaluations", json={"plant_id": plant["id"], "episodes": 1}).json()
    assert set(evaluation["results"]) == {"pull", "fifo", "random"}


def test_training_job(tmp_path):
    training = {"timesteps": 256, "n_envs": 1, "n_steps": 128, "batch_size": 64, "eval_freq": 128}
    training |= {"pretrain_episodes": 2, "pretrain_epochs": 2}
    with TestClient(create_app(tmp_path)) as client:
        plant_id = client.get("/plants").json()[0]["id"]
        job = client.post("/training-jobs", json={"plant_id": plant_id, "training": training}).json()
        deadline = time.time() + 300
        while (job := client.get(f"/training-jobs/{job['id']}").json())["status"] in ("queued", "running"):
            assert time.time() < deadline
            time.sleep(1)
        assert job["status"] == "completed", job["error"]
        assert job["curve"]

        sim = client.post("/simulations", json={"policy": "model", "model_id": job["model_id"]})
        assert sim.status_code == 201
        download = client.get(f"/models/{job['model_id']}/download")
        assert download.headers["content-type"] == "application/zip"
