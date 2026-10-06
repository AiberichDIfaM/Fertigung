"""HTTP routes for the shop-floor model: shop configurations, validation and simulations."""

from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fertigung.api.store import Store, now
from fertigung.shop.config import PROFILES, ShopConfig
from fertigung.shop.model import ShopModel
from fertigung.shop.report import POLICIES, simulate, workshop_config
from fertigung.shop.validation import validate_shop


class ShopSimulationCreate(BaseModel):
    shop_id: str | None = Field(None, description="Stored shop; alternatively pass `config`")
    config: ShopConfig | None = None
    policy: Literal["pull"] = "pull"
    minutes: int | None = Field(
        None, ge=1, description="Default: one day after the last deadline, at least a week"
    )
    queue_limit: int = Field(2, ge=1, description="Planned but not started jobs per machine")


def seed_shop(store: Store):
    if store.get_setting("seeded_shop"):
        return
    if not store.list("shops"):
        config = workshop_config()
        store.insert("shops", name=config.name, config=config.model_dump(by_alias=True), updated_at=now())
    store.set_setting("seeded_shop", now())


def analyze_shop(config: ShopConfig) -> dict:
    model = ShopModel(config)
    families = {
        mt.name: sorted(
            {t.setup_family for t in config.transformations if t.name in mt.transformations} - {None}
        )
        for mt in config.machine_types
    }
    return {
        "issues": [asdict(i) for i in validate_shop(config)],
        "raw": sorted(model.raw),
        "final": sorted(model.final),
        "intermediate": sorted(set(model.parts) - model.raw - model.final),
        "families": families,
    }


def shop_router(store: Store, dependencies) -> APIRouter:
    router = APIRouter(dependencies=dependencies)

    def get_or_404(table: str, id: str) -> dict:
        row = store.get(table, id)
        if row is None:
            raise HTTPException(404, f"{table[:-1]} {id} not found")
        return row

    @router.get("/shops")
    def list_shops():
        return store.list("shops")

    @router.post("/shops", status_code=201)
    def create_shop(config: ShopConfig):
        return store.insert(
            "shops", name=config.name, config=config.model_dump(by_alias=True), updated_at=now()
        )

    @router.post("/shops/validate")
    def validate_config(config: ShopConfig):
        return analyze_shop(config)

    @router.get("/shops/{shop_id}")
    def get_shop(shop_id: str):
        return get_or_404("shops", shop_id)

    @router.put("/shops/{shop_id}")
    def update_shop(shop_id: str, config: ShopConfig):
        get_or_404("shops", shop_id)
        return store.update(
            "shops", shop_id, name=config.name, config=config.model_dump(by_alias=True), updated_at=now()
        )

    @router.delete("/shops/{shop_id}", status_code=204)
    def delete_shop(shop_id: str):
        if not store.delete("shops", shop_id):
            raise HTTPException(404, f"shop {shop_id} not found")

    @router.post("/shops/{shop_id}/validate")
    def validate_shop_by_id(shop_id: str):
        return analyze_shop(ShopConfig.model_validate(get_or_404("shops", shop_id)["config"]))

    @router.get("/shop-profiles")
    def shop_profiles():
        return PROFILES

    @router.get("/shop-policies")
    def shop_policies():
        return sorted(POLICIES)

    @router.post("/shop-simulations", status_code=201)
    def create_simulation(body: ShopSimulationCreate):
        if body.config is not None:
            config = body.config
        elif body.shop_id:
            config = ShopConfig.model_validate(get_or_404("shops", body.shop_id)["config"])
        else:
            raise HTTPException(422, "pass shop_id or config")
        if any(i.level == "error" for i in validate_shop(config)):
            raise HTTPException(422, "shop has validation errors")
        result = simulate(config, body.policy, body.minutes, body.queue_limit)
        request = body.model_dump(exclude={"config"}) | {"shop": config.name}
        return store.insert("shop_simulations", request=request, result=result)

    @router.get("/shop-simulations/{simulation_id}")
    def get_simulation(simulation_id: str):
        return get_or_404("shop_simulations", simulation_id)

    return router
