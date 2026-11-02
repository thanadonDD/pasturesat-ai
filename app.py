"""
GrassSense x Pasture.io — Full-Stack Satellite Pasture Intelligence Server
FastAPI Backend serving:
- /api/farm : Full multi-paddock satellite + ML analysis, Pasture.io Feed Wedge, Herd Grazing Plan, SDG summary
- /api/calculate : Direct interactive calculator for custom Sentinel-2 10 bands + Sentinel-1 SAR + Weather -> 15 VIs + SVR/RF DM, CP, NDF, ADF, PGR
- /api/paddock/add : Add a custom user-drawn GeoJSON polygon paddock on the map and fetch real STAC + Open-Meteo data
- / : Serves the Pasture.io x GrassSense web dashboard
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.ml_models import GLOBAL_ML_ENGINE, compute_vegetation_indices, evaluate_grazing_readiness
from backend.satellite_engine import DEFAULT_PADDOCKS, analyze_paddock_complete, polygon_area_and_centroid
from backend.cog_reader import extract_polygon_real_pixels
from backend.pasture_io_suite import (
    DEFAULT_MOBS,
    ACTIVE_MOBS,
    DEFAULT_ACTIVITIES,
    ACTIVE_ACTIVITIES,
    PADDOCK_MODIFIERS,
    apply_paddock_modifiers,
    record_paddock_activity,
    build_pasture_io_suite,
)
from backend.ground_truth_validator import (
    run_ground_truth_validation_suite,
    evaluate_single_lab_ground_truth,
    extract_live_proof_at_latlon,
)

app = FastAPI(
    title="GrassSense x Pasture.io — Satellite AI Pasture Intelligence",
    description="Sentinel-1/2 + Machine Learning (RF & SVR) Pasture Biomass (DM), Crude Protein (CP), NDF/ADF & Rotational Grazing Decision Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# In-memory store initialized with our 6 default paddocks
ACTIVE_PADDOCKS: List[Dict[str, Any]] = [dict(p) for p in DEFAULT_PADDOCKS]


class CustomCalculateRequest(BaseModel):
    paddock_name: str = "แปลงทดสอบ (Custom Lab)"
    species: str = "giant_napier"
    area_rai: float = 12.5
    rest_days: float = 38.0
    herd_count: int = 20
    cow_weight_kg: float = 350.0
    cloud_cover_pct: float = 8.0
    bands: Dict[str, float] = {
        "B2": 0.036,
        "B3": 0.068,
        "B4": 0.042,
        "B5": 0.098,
        "B6": 0.258,
        "B7": 0.332,
        "B8": 0.375,
        "B8A": 0.398,
        "B11": 0.168,
        "B12": 0.092,
    }
    sar: Dict[str, float] = {"VH": -13.6, "VV": -8.2}
    weather: Dict[str, float] = {
        "Tmax": 33.2,
        "Tmin": 23.6,
        "Tavg": 28.4,
        "RH": 74.0,
        "ND": 12,
        "Rainfall": 135.0,
    }


class AddPaddockRequest(BaseModel):
    name: str
    species: str = "giant_napier"
    coordinates: List[List[float]]


class AddActivityRequest(BaseModel):
    paddock_id: str
    activity_type: str = "FERTILIZER"
    amount_value: float = 25.0
    detail: str = ""
    mob_id: Optional[str] = None


class MoveMobRequest(BaseModel):
    mob_id: str
    target_paddock_id: str
    head_count: Optional[int] = None
    supplement_kg_dm_head: Optional[float] = None


class CheckLabGroundTruthRequest(BaseModel):
    paddock_id: str = "PADDOCK-A1"
    lab_dm_kg_ha: float = 3480.0
    lab_cp_pct: float = 15.9
    lab_ndf_pct: float = 60.4


class LivePixelProofRequest(BaseModel):
    lat: float = -21.24073
    lon: float = -48.29949
    scene_id: str = "S2B_22KGB_20200214_1_L2A"
    species: str = "mombasa_guinea"
    rest_days: float = 36.0
    lab_dm_kg_ha: float = 3350.0
    lab_cp_pct: float = 16.2
    lab_ndf_pct: float = 58.5


@app.get("/api/farm")
def get_farm_overview(
    herd_count: int = 20,
    cow_weight_kg: float = 350.0,
    target_pre_grazing: float = 2800.0,
    target_residual: float = 1500.0,
) -> Dict[str, Any]:
    """
    Analyzes all paddocks in the farm, constructs the Pasture.io Feed Wedge,
    computes farm-wide totals, and returns satellite STAC + Open-Meteo + full Pasture.io Suite results.
    """
    if ACTIVE_MOBS:
        ACTIVE_MOBS[0]["head_count"] = herd_count
        ACTIVE_MOBS[0]["avg_weight_kg"] = cow_weight_kg

    analyzed = [
        apply_paddock_modifiers(analyze_paddock_complete(p, herd_count=herd_count, cow_weight_kg=cow_weight_kg))
        for p in ACTIVE_PADDOCKS
    ]

    # Sort paddocks by green grazeable DM (kg DM/ha) descending to build the Pasture.io Feed Wedge
    feed_wedge = sorted(
        [
            {
                "id": p["id"],
                "name": p["name"],
                "area_rai": p["area_rai"],
                "area_ha": p["area_ha"],
                "green_dm_kg_ha": p["prediction"]["biomass"]["green_dm_kg_ha"],
                "total_dm_kg_rai": p["prediction"]["biomass"]["total_dm_kg_rai"],
                "available_dm_kg": p["readiness"]["grazing_plan"]["total_available_dm_kg"],
                "grazing_days": p["readiness"]["grazing_plan"]["grazing_days_available"],
                "cp_pct": p["prediction"]["nutrition"]["cp_pct"],
                "ndf_pct": p["prediction"]["nutrition"]["ndf_pct"],
                "age_days": p["prediction"]["maturity"]["estimated_age_days"],
                "pgr_kg_ha_day": p["prediction"]["biomass"]["pgr_kg_ha_day"],
                "NDVI": p["prediction"]["vegetation_indices"]["NDVI"],
                "NDRE": p["prediction"]["vegetation_indices"]["NDRE"],
                "CCCI": p["prediction"]["vegetation_indices"]["CCCI"],
                "status_code": p["readiness"]["overall_code"],
                "status_badge": p["readiness"]["overall_badge"],
                "status_color": p["readiness"]["overall_color"],
            }
            for p in analyzed
        ],
        key=lambda x: x["green_dm_kg_ha"],
        reverse=True,
    )

    valid_pastures = [
        p for p in analyzed
        if p.get("is_valid_pasture", True)
        and p["readiness"]["overall_code"] not in ("REST_BARE_OR_DRY_SOIL", "NOT_PASTURE_LAND")
    ]
    if not valid_pastures:
        valid_pastures = [p for p in analyzed if p.get("is_valid_pasture", True)]
    denom = max(1, len(valid_pastures))

    total_area_rai = round(sum(p["area_rai"] for p in valid_pastures), 2)
    total_area_ha = round(sum(p["area_ha"] for p in valid_pastures), 2)
    avg_ndvi = round(sum(p["prediction"]["vegetation_indices"]["NDVI"] for p in valid_pastures) / denom, 3)
    avg_cp = round(sum(p["prediction"]["nutrition"]["cp_pct"] for p in valid_pastures) / denom, 2)
    avg_ndf = round(sum(p["prediction"]["nutrition"]["ndf_pct"] for p in valid_pastures) / denom, 2)
    avg_dm_ha = round(sum(p["prediction"]["biomass"]["green_dm_kg_ha"] for p in valid_pastures) / denom, 0)
    avg_pgr = round(sum(p["prediction"]["biomass"]["pgr_kg_ha_day"] for p in valid_pastures) / denom, 1)
    total_grazing_days = round(sum(p["readiness"]["grazing_plan"]["grazing_days_available"] for p in analyzed if p["readiness"]["overall_code"] in ("READY_TO_GRAZE", "ALMOST_READY")), 1)
    ready_count = sum(1 for p in analyzed if p["readiness"]["overall_code"] == "READY_TO_GRAZE")

    pio_suite = build_pasture_io_suite(
        analyzed_paddocks=analyzed,
        target_pre_grazing_kg_ha=target_pre_grazing,
        target_residual_kg_ha=target_residual,
    )
    gt_validation = run_ground_truth_validation_suite()

    return {
        "farm_name": "GrassSense Smart Ranch (ปากช่อง จ.นครราชสีมา)",
        "herd_count": herd_count,
        "cow_weight_kg": cow_weight_kg,
        "summary": {
            "total_paddocks": len(analyzed),
            "ready_paddocks": ready_count,
            "total_area_rai": total_area_rai,
            "total_area_ha": total_area_ha,
            "avg_ndvi": avg_ndvi,
            "avg_cp_pct": avg_cp,
            "avg_ndf_pct": avg_ndf,
            "avg_green_dm_kg_ha": avg_dm_ha,
            "avg_pgr_kg_ha_day": avg_pgr,
            "rotation_grazing_days_ready": total_grazing_days,
        },
        "feed_wedge": feed_wedge,
        "pasture_io_suite": pio_suite,
        "ground_truth_validation": gt_validation,
        "paddocks": analyzed,
    }


@app.post("/api/calculate")
def calculate_custom_spectral(req: CustomCalculateRequest) -> Dict[str, Any]:
    """
    Runs the full satellite indices + ML inference + GrassSense 4-variable readiness check
    on custom Sentinel-2 bands, Sentinel-1 SAR, and meteorological inputs.
    """
    pred = GLOBAL_ML_ENGINE.predict_all(
        bands=req.bands,
        weather=req.weather,
        sar=req.sar,
        grass_species=req.species,
        days_since_cut=req.rest_days,
    )
    valid_pixels = max(10, round(req.area_rai * 16.0))
    readiness = evaluate_grazing_readiness(
        prediction=pred,
        area_rai=req.area_rai,
        herd_count=req.herd_count,
        cow_weight_kg=req.cow_weight_kg,
        cloud_cover_pct=req.cloud_cover_pct,
        valid_pixels=valid_pixels,
        data_age_days=2,
        paddock_name=req.paddock_name,
    )
    return {
        "paddock_name": req.paddock_name,
        "area_rai": req.area_rai,
        "prediction": pred,
        "readiness": readiness,
    }


@app.post("/api/paddock/add")
def add_custom_paddock(req: AddPaddockRequest) -> Dict[str, Any]:
    """Adds a user-drawn polygon paddock on the map and extracts real 10m×10m Sentinel-2 COG pixels from AWS S3."""
    existing_ids = {p["id"] for p in ACTIVE_PADDOCKS}
    idx_num = 1
    while f"PADDOCK-U{idx_num}" in existing_ids:
        idx_num += 1
    new_id = f"PADDOCK-U{idx_num}"
    p_name = req.name.strip() if req.name else f"แปลงวาดใหม่ #{idx_num}"
    if p_name == "แปลงทดสอบ (วาดใหม่)":
        p_name = f"แปลงทดสอบ #{idx_num} (วาดใหม่)"
    new_paddock: Dict[str, Any] = {
        "id": new_id,
        "name": p_name,
        "region": "วิเคราะห์จากพิกเซลดาวเทียม Sentinel-2 L2A จริง (10×10m)",
        "species": req.species,
        "coordinates": req.coordinates,
        "is_custom": True,
    }
    ACTIVE_PADDOCKS.append(new_paddock)
    return analyze_paddock_complete(new_paddock)


@app.delete("/api/paddock/{paddock_id}")
def delete_paddock(paddock_id: str) -> Dict[str, Any]:
    """Deletes a specific paddock by ID from ACTIVE_PADDOCKS."""
    global ACTIVE_PADDOCKS
    before_len = len(ACTIVE_PADDOCKS)
    ACTIVE_PADDOCKS = [p for p in ACTIVE_PADDOCKS if p["id"] != paddock_id]
    # If all paddocks were deleted, restore the first default paddock so the UI always has at least one plot
    if not ACTIVE_PADDOCKS:
        ACTIVE_PADDOCKS = [dict(DEFAULT_PADDOCKS[0])]
    return {
        "status": "deleted" if len(ACTIVE_PADDOCKS) < before_len else "not_found",
        "deleted_id": paddock_id,
        "remaining_count": len(ACTIVE_PADDOCKS),
    }


@app.post("/api/paddock/reset")
def reset_paddocks() -> Dict[str, Any]:
    """Clears all user-drawn paddocks, activities, and modifiers, restoring defaults."""
    global ACTIVE_PADDOCKS
    ACTIVE_PADDOCKS = [dict(p) for p in DEFAULT_PADDOCKS]
    PADDOCK_MODIFIERS.clear()
    ACTIVE_ACTIVITIES.clear()
    ACTIVE_ACTIVITIES.extend([dict(a) for a in DEFAULT_ACTIVITIES])
    ACTIVE_MOBS.clear()
    ACTIVE_MOBS.extend([dict(m) for m in DEFAULT_MOBS])
    return {"status": "reset", "count": len(ACTIVE_PADDOCKS)}


@app.post("/api/activity/add")
def add_paddock_activity(req: AddActivityRequest) -> Dict[str, Any]:
    """Records a Pasture.io Paddock Activity (Grazing, Fertiliser, Harvest, Irrigation, Plate Meter) and updates paddock state."""
    p_obj = next((p for p in ACTIVE_PADDOCKS if p["id"] == req.paddock_id), None)
    p_name = p_obj["name"].split(" - ")[0] if p_obj else req.paddock_id
    act = record_paddock_activity(
        paddock_id=req.paddock_id,
        paddock_name=p_name,
        activity_type=req.activity_type,
        amount_value=req.amount_value,
        detail=req.detail,
        mob_id=req.mob_id,
    )
    return {"status": "logged", "activity": act}


@app.post("/api/mob/move")
def move_mob_to_paddock(req: MoveMobRequest) -> Dict[str, Any]:
    """Moves a livestock mob to a target paddock and optionally updates head count or supplement rate."""
    for m in ACTIVE_MOBS:
        if m["mob_id"] == req.mob_id:
            m["current_paddock_id"] = req.target_paddock_id
            if req.head_count is not None and req.head_count > 0:
                m["head_count"] = req.head_count
            if req.supplement_kg_dm_head is not None and req.supplement_kg_dm_head >= 0:
                m["supplement_kg_dm_head"] = req.supplement_kg_dm_head
            return {"status": "moved", "mob": m}
    return {"status": "not_found"}


@app.post("/api/validation/check-lab")
def check_lab_ground_truth(req: CheckLabGroundTruthRequest) -> Dict[str, Any]:
    """Compares user-entered Laboratory Ground Truth (DM, CP%, NDF%) against a paddock's satellite prediction."""
    p_obj = next((p for p in ACTIVE_PADDOCKS if p["id"] == req.paddock_id), ACTIVE_PADDOCKS[0])
    analyzed_p = apply_paddock_modifiers(analyze_paddock_complete(p_obj))
    res = evaluate_single_lab_ground_truth(
        paddock_pred=analyzed_p["prediction"],
        lab_dm_kg_ha=req.lab_dm_kg_ha,
        lab_cp_pct=req.lab_cp_pct,
        lab_ndf_pct=req.lab_ndf_pct,
    )
    return {
        "paddock_id": analyzed_p["id"],
        "paddock_name": analyzed_p["name"],
        "evaluation": res,
    }


@app.post("/api/validation/live-proof")
def live_pixel_ground_truth_proof(req: LivePixelProofRequest) -> Dict[str, Any]:
    """
    Extracts the real uint16 GeoTIFF pixel (B4_dn, B5_dn, B8_dn, B11_dn) and 3x3 neighborhood
    directly from the AWS S3 Sentinel-2 L2A tile at ANY user-clicked (lat, lon) on UNESP Brazil (22KGB)
    or Pak Chong Thailand (47PQS), runs the AI models, and evaluates against Lab Ground Truth.
    """
    return extract_live_proof_at_latlon(
        lat=req.lat,
        lon=req.lon,
        scene_id=req.scene_id,
        species=req.species,
        rest_days=req.rest_days,
        lab_dm_kg_ha=req.lab_dm_kg_ha,
        lab_cp_pct=req.lab_cp_pct,
        lab_ndf_pct=req.lab_ndf_pct,
    )


@app.get("/api/real-cog")
def get_real_cog_pixels(lat: float = 14.6530, lon: float = 101.3990) -> Dict[str, Any]:
    """Directly reads and returns real Sentinel-2 L2A B02..B12 COG pixel values from AWS S3 at (lat, lon)."""
    coords = [
        [lat + 0.0005, lon - 0.0005],
        [lat + 0.0005, lon + 0.0005],
        [lat - 0.0005, lon + 0.0005],
        [lat - 0.0005, lon - 0.0005],
    ]
    res = extract_polygon_real_pixels(coords)
    return res or {"error": "Could not read COG tile"}


STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )

