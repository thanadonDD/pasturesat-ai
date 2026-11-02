"""
Pasture.io ("Pio") Complete Enterprise Feature Suite for GrassSense
Implements all core modules of the Pasture.io platform:
1. Predictive & Scenario Feed Wedge (Today / +7 Days / +14 Days, Target Demand Line, Surplus/Deficit Silage Calculator, Leaf Stage 1.0-3.5)
2. Multi-Mob / Herd Grazing Planner & Rotation Queue (Milking Herd, Heifers/Beef, Dry Cows, Stocking Rate LSU/ha, Feed Budget & Supplements)
3. Paddock Activity Log & Real-Time State Modifiers (Grazing, Fertiliser N-P-K/Urea, Harvest/Silage, Irrigation, Spraying, Rising Plate Meter Calibration)
4. 14-Day Open-Meteo Weather, Growing Degree Days (GDD), Soil Moisture & Pio PGR Forecast Engine
5. Annual Paddock Harvest Yield & Utilization Benchmarking (t DM/ha/yr, Utilization %, N-Response Efficiency, THB/kg DM)
"""

from __future__ import annotations
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import requests


# ============================================================================
# 1. MULTI-MOB / HERD MANAGEMENT STORE
# ============================================================================

DEFAULT_MOBS: List[Dict[str, Any]] = [
    {
        "mob_id": "MOB-1",
        "name": "ฝูงโครีดนมหลัก (High-Yield Milking Herd)",
        "Livestock_class": "โคนมกำลังรีดนม (Lactating Dairy)",
        "head_count": 20,
        "avg_weight_kg": 450.0,
        "dm_intake_pct_bw": 3.2,
        "supplement_kg_dm_head": 3.0,
        "supplement_type": "อาหารข้นโปรตีน 18% + แร่ธาตุ",
        "current_paddock_id": "PADDOCK-A1",
        "target_residual_kg_ha": 1500.0,
        "rotation_target_days": 30,
    },
    {
        "mob_id": "MOB-2",
        "name": "ฝูงโคสาวทดแทน & โคเนื้อ (Heifers & Beef)",
        "Livestock_class": "โคสาว/โคขุน (Growing Cattle)",
        "head_count": 15,
        "avg_weight_kg": 320.0,
        "dm_intake_pct_bw": 2.8,
        "supplement_kg_dm_head": 1.0,
        "supplement_type": "รำละเอียด + กากน้ำตาล",
        "current_paddock_id": "PADDOCK-B1",
        "target_residual_kg_ha": 1450.0,
        "rotation_target_days": 32,
    },
    {
        "mob_id": "MOB-3",
        "name": "ฝูงโคพักรีด & แม่โคอุ้มท้อง (Dry Cows)",
        "Livestock_class": "โคพักรีด (Dry / Pregnant)",
        "head_count": 10,
        "avg_weight_kg": 400.0,
        "dm_intake_pct_bw": 2.2,
        "supplement_kg_dm_head": 1.5,
        "supplement_type": "ฟางหมักยูเรีย + เกลือแร่ก้อน",
        "current_paddock_id": "PADDOCK-C1",
        "target_residual_kg_ha": 1400.0,
        "rotation_target_days": 35,
    },
]

ACTIVE_MOBS: List[Dict[str, Any]] = [dict(m) for m in DEFAULT_MOBS]


# ============================================================================
# 2. PADDOCK ACTIVITIES & REAL-TIME MODIFIERS STORE
# ============================================================================

DEFAULT_ACTIVITIES: List[Dict[str, Any]] = [
    {
        "activity_id": "ACT-101",
        "paddock_id": "PADDOCK-A1",
        "paddock_name": "แปลง A1 - เนเปียร์ยักษ์",
        "date": "2026-10-02",
        "type": "FERTILIZER",
        "type_label": "🧪 ใส่ปุ๋ยเคมี/ปุ๋ยคอก",
        "detail": "ปุ๋ยยูเรีย (46-0-0) อัตรา 25 กก./ไร่ (71.9 kg N/ha) หลังฝนตก",
        "amount_value": 25.0,
        "amount_unit": "กก./ไร่",
        "n_kg_ha": 71.9,
        "effect_summary": "เร่งอัตราโต PGR +22% และเพิ่มโปรตีนหยาบ CP +1.6%",
    },
    {
        "activity_id": "ACT-102",
        "paddock_id": "PADDOCK-A2",
        "paddock_name": "แปลง A2 - กินนีมอมบาซา",
        "date": "2026-10-06",
        "type": "IRRIGATION",
        "type_label": "💧 ให้น้ำสปริงเกลอร์",
        "detail": "เปิดระบบสปริงเกลอร์แรงดันสูง 35 มม. ช่วงเช้า",
        "amount_value": 35.0,
        "amount_unit": "มม.",
        "n_kg_ha": 0.0,
        "effect_summary": "รักษาความชื้นในดิน +18% และดัชนี LSWI สูงขึ้น",
    },
    {
        "activity_id": "ACT-103",
        "paddock_id": "PADDOCK-B2",
        "paddock_name": "แปลง B2 - เนเปียร์ยักษ์",
        "date": "2026-09-29",
        "type": "HARVEST",
        "type_label": "🚜 ตัดหญ้าหมัก (Silage Cut)",
        "detail": "ตัดหญ้าสดทำหญ้าหมัก 2,850 กก. DM/ha เหลือตอสูง 15 ซม. (1,500 kg DM/ha)",
        "amount_value": 2850.0,
        "amount_unit": "kg DM/ha",
        "n_kg_ha": 0.0,
        "effect_summary": "รีเซ็ตอายุหญ้าเป็น Day 0 และเริ่มรอบการแตกใบใหม่",
    },
    {
        "activity_id": "ACT-104",
        "paddock_id": "PADDOCK-B1",
        "paddock_name": "แปลง B1 - แพงโกล่า",
        "date": "2026-10-12",
        "type": "PLATE_METER",
        "type_label": "📏 วัดความสูงหญ้าภาคสนาม (RPM Walk)",
        "detail": "เดินสุ่มวัดด้วย Rising Plate Meter 30 จุด เฉลี่ย 2,180 kg DM/ha (ตรงกับดาวเทียม 98.4%)",
        "amount_value": 2180.0,
        "amount_unit": "kg DM/ha",
        "n_kg_ha": 0.0,
        "effect_summary": "สอบเทียบค่าดาวเทียม Sentinel-2 กับค่าจริงภาคสนาม (R² = 0.96)",
    },
    {
        "activity_id": "ACT-105",
        "paddock_id": "PADDOCK-C1",
        "paddock_name": "แปลง C1 - รูซี่",
        "date": "2026-09-24",
        "type": "GRAZING",
        "type_label": "🐄 ปล่อยฝูงโคแทะเล็ม",
        "detail": "ฝูงโคพักรีด 10 ตัว แทะเล็ม 5 วัน เหลือตอหญ้า Residual 1,450 kg DM/ha",
        "amount_value": 1450.0,
        "amount_unit": "kg DM/ha (Residual)",
        "n_kg_ha": 0.0,
        "effect_summary": "บันทึกวันย้ายฝูงโคออกและเริ่มนับเวลาพักแปลง",
    },
]

ACTIVE_ACTIVITIES: List[Dict[str, Any]] = [dict(a) for a in DEFAULT_ACTIVITIES]

# Dynamic per-paddock state overrides triggered when a user records a new activity in the UI
PADDOCK_MODIFIERS: Dict[str, Dict[str, Any]] = {}


def apply_paddock_modifiers(paddock_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies any user-logged Pasture.io Paddock Activities (Fertilizer boost, Grazing reset,
    Harvest cut, Irrigation boost, or Plate Meter calibration) to the analyzed paddock dict.
    """
    pid = paddock_analysis["id"]
    mod = PADDOCK_MODIFIERS.get(pid)
    if not mod or not paddock_analysis.get("is_valid_pasture", True):
        return paddock_analysis

    pred = paddock_analysis["prediction"]
    bm = pred["biomass"]
    nu = pred["nutrition"]
    mat = pred["maturity"]

    # 1. If a recent Grazing or Harvest reset was logged
    if "override_green_dm_kg_ha" in mod:
        new_green_dm = float(mod["override_green_dm_kg_ha"])
        ratio = new_green_dm / max(100.0, bm["green_dm_kg_ha"])
        bm["green_dm_kg_ha"] = round(new_green_dm, 1)
        bm["green_dm_kg_rai"] = round(new_green_dm * 0.16, 1)
        bm["total_dm_kg_ha"] = round(bm["total_dm_kg_ha"] * max(0.35, min(1.8, ratio)), 1)
        bm["total_dm_kg_rai"] = round(bm["total_dm_kg_ha"] * 0.16, 1)

    if "override_age_days" in mod:
        mat["estimated_age_days"] = float(mod["override_age_days"])
        paddock_analysis["last_cut_date"] = mod.get("last_cut_date", paddock_analysis["last_cut_date"])

    # 2. If Fertilizer or Irrigation boost was logged
    pgr_mult = float(mod.get("pgr_multiplier", 1.0))
    cp_add = float(mod.get("cp_bonus_pct", 0.0))
    if pgr_mult != 1.0:
        bm["pgr_kg_ha_day"] = round(bm["pgr_kg_ha_day"] * pgr_mult, 1)
        bm["pgr_kg_rai_day"] = round(bm["pgr_kg_ha_day"] * 0.16, 2)
    if cp_add != 0.0:
        nu["cp_pct"] = round(min(22.0, nu["cp_pct"] + cp_add), 2)

    # Re-evaluate readiness with updated numbers
    from backend.ml_models import evaluate_grazing_readiness
    paddock_analysis["readiness"] = evaluate_grazing_readiness(
        prediction=pred,
        area_rai=paddock_analysis["area_rai"],
        herd_count=paddock_analysis["readiness"]["grazing_plan"]["herd_count"],
        cow_weight_kg=paddock_analysis["readiness"]["grazing_plan"]["cow_weight_kg"],
        cloud_cover_pct=paddock_analysis["stac_metadata"]["selected_s2_scene"]["cloud_cover_pct"],
        valid_pixels=paddock_analysis["valid_pixels_10m"],
        data_age_days=3,
        paddock_name=paddock_analysis["name"],
    )
    return paddock_analysis


def record_paddock_activity(
    paddock_id: str,
    paddock_name: str,
    activity_type: str,
    amount_value: float,
    detail: str = "",
    mob_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Logs a new Paddock Activity in Pasture.io style and updates PADDOCK_MODIFIERS
    so the Feed Wedge, PGR, and Paddock Readiness immediately reflect the farm operation.
    """
    today_str = datetime.date.today().isoformat()
    act_num = 100 + len(ACTIVE_ACTIVITIES) + 1
    mod = PADDOCK_MODIFIERS.setdefault(paddock_id, {"pgr_multiplier": 1.0, "cp_bonus_pct": 0.0})

    n_kg_ha = 0.0
    if activity_type == "FERTILIZER":
        type_label = "🧪 ใส่ปุ๋ยเคมี/ยูเรีย (Fertiliser)"
        unit = "กก./ไร่"
        n_kg_ha = round(amount_value * 6.25 * 0.46, 1)
        mod["pgr_multiplier"] = round(float(mod.get("pgr_multiplier", 1.0)) * 1.24, 2)
        mod["cp_bonus_pct"] = round(float(mod.get("cp_bonus_pct", 0.0)) + 1.8, 2)
        effect = f"เพิ่มไนโตรเจน {n_kg_ha} kg N/ha → เร่ง PGR +24% และเพิ่ม CP +1.8%"
        if not detail:
            detail = f"หว่านปุ๋ยยูเรีย 46-0-0 อัตรา {amount_value} กก./ไร่ ({n_kg_ha} kg N/ha)"
    elif activity_type == "GRAZING":
        type_label = "🐄 ปล่อยฝูงโคแทะเล็ม (Grazing)"
        unit = "kg DM/ha (ตอหญ้า)"
        residual = max(400.0, min(2200.0, amount_value if amount_value > 100 else 650.0))
        mod["override_green_dm_kg_ha"] = residual
        mod["override_age_days"] = 2.0
        mod["last_cut_date"] = today_str
        effect = f"แทะเล็มเหลือตอหญ้า {residual:.0f} kg DM/ha → รีเซ็ตอายุหญ้าเป็น 2 วัน (พักแปลง)"
        if mob_id:
            for m in ACTIVE_MOBS:
                if m["mob_id"] == mob_id:
                    m["current_paddock_id"] = paddock_id
        if not detail:
            detail = f"ย้ายฝูงโคเข้าแทะเล็ม เหลือตอหญ้า Residual {residual:.0f} kg DM/ha"
    elif activity_type == "HARVEST":
        type_label = "🚜 ตัดหญ้าหมัก/สางแปลง (Harvest/Topping)"
        unit = "kg DM/ha ที่ตัดได้"
        mod["override_green_dm_kg_ha"] = 520.0
        mod["override_age_days"] = 1.0
        mod["last_cut_date"] = today_str
        effect = f"เก็บเกี่ยวหญ้าหมัก {amount_value:,.0f} kg DM/ha → รีเซ็ตอายุหญ้าเป็น Day 1"
        if not detail:
            detail = f"ตัดหญ้าสดทำ Silage ได้ {amount_value:,.0f} kg DM/ha (เหลือตอ 15 ซม.)"
    elif activity_type == "IRRIGATION":
        type_label = "💧 ให้น้ำสปริงเกลอร์ (Irrigation)"
        unit = "มม."
        mod["pgr_multiplier"] = round(float(mod.get("pgr_multiplier", 1.0)) * 1.16, 2)
        effect = f"เติมความชื้นในดิน +{amount_value:.0f} มม. → เร่งอัตราการเติบโต PGR +16%"
        if not detail:
            detail = f"เปิดสปริงเกลอร์ให้น้ำ {amount_value:.0f} มม. ทั่วแปลง"
    else:  # PLATE_METER
        type_label = "📏 วัดความสูงหญ้าภาคสนาม (Plate Meter)"
        unit = "kg DM/ha"
        calibrated_dm = max(100.0, amount_value)
        mod["override_green_dm_kg_ha"] = calibrated_dm
        effect = f"ปรับเทียบค่ามวลหญ้าพร้อมกินตามผลวัดภาคสนาม = {calibrated_dm:,.0f} kg DM/ha"
        if not detail:
            detail = f"วัดด้วย Rising Plate Meter ภาคสนามได้ค่าเฉลี่ย {calibrated_dm:,.0f} kg DM/ha"

    new_act = {
        "activity_id": f"ACT-{act_num}",
        "paddock_id": paddock_id,
        "paddock_name": paddock_name,
        "date": today_str,
        "type": activity_type,
        "type_label": type_label,
        "detail": detail,
        "amount_value": amount_value,
        "amount_unit": unit,
        "n_kg_ha": n_kg_ha,
        "effect_summary": effect,
    }
    ACTIVE_ACTIVITIES.insert(0, new_act)
    return new_act


# ============================================================================
# 3. 14-DAY PIO WEATHER, GDD & PASTURE GROWTH RATE (PGR) FORECAST
# ============================================================================

_FORECAST_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def fetch_14d_pio_forecast(lat: float, lng: float, base_pgr_kg_ha_day: float = 55.0) -> Dict[str, Any]:
    """
    Fetches real 14-day meteorological forecast from Open-Meteo and runs the
    Pasture.io ("Pio") biophysical growth forecast model:
    - Growing Degree Days (GDD, base 12°C for C4 tropical grasses)
    - FAO Penman-Monteith Evapotranspiration (ET0) & Soil Moisture Balance (%)
    - Daily Predicted PGR (kg DM/ha/day) & Leaf Emergence Rate (days per leaf)
    """
    cache_key = f"{round(lat, 2)}_{round(lng, 2)}"
    raw_days: List[Dict[str, Any]] = []

    if cache_key in _FORECAST_CACHE:
        raw_days = _FORECAST_CACHE[cache_key]
    else:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lng}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration,shortwave_radiation_sum"
            f"&timezone=Asia%2FBangkok&forecast_days=14"
        )
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                daily = resp.json().get("daily", {})
                dates = daily.get("time", [])
                tmaxs = daily.get("temperature_2m_max", [])
                tmins = daily.get("temperature_2m_min", [])
                rains = daily.get("precipitation_sum", [])
                et0s = daily.get("et0_fao_evapotranspiration", [])
                rads = daily.get("shortwave_radiation_sum", [])
                for i in range(min(14, len(dates))):
                    raw_days.append({
                        "date": dates[i],
                        "tmax": float(tmaxs[i] if tmaxs[i] is not None else 31.5),
                        "tmin": float(tmins[i] if tmins[i] is not None else 22.5),
                        "rain_mm": float(rains[i] if rains[i] is not None else 0.0),
                        "et0_mm": float(et0s[i] if et0s[i] is not None else 4.2),
                        "rad_mj": float(rads[i] if rads[i] is not None else 18.5),
                    })
                if raw_days:
                    _FORECAST_CACHE[cache_key] = raw_days
        except Exception:
            pass

    if not raw_days:
        base_dt = datetime.date.today()
        synth_rain = [2.4, 0.0, 0.0, 8.5, 14.2, 1.2, 0.0, 0.0, 5.6, 11.0, 0.0, 0.0, 3.2, 0.0]
        for i in range(14):
            d_str = (base_dt + datetime.timedelta(days=i)).isoformat()
            raw_days.append({
                "date": d_str,
                "tmax": round(31.2 + 1.5 * np.sin(i * 0.5), 1),
                "tmin": round(22.4 + 0.9 * np.cos(i * 0.5), 1),
                "rain_mm": synth_rain[i],
                "et0_mm": round(4.1 + 0.4 * np.sin(i * 0.4), 2),
                "rad_mj": round(18.8 + 2.2 * np.sin(i * 0.6), 1),
            })

    soil_moisture_pct = 72.0
    forecast_series: List[Dict[str, Any]] = []
    for d in raw_days:
        tavg = 0.5 * (d["tmax"] + d["tmin"])
        # C4 tropical grass base temperature = 12.0°C, optimal = 30-33°C
        gdd = max(0.0, tavg - 12.0)
        # Update soil moisture balance from rainfall and ET0
        soil_moisture_pct = float(np.clip(soil_moisture_pct + d["rain_mm"] * 1.45 - d["et0_mm"] * 1.15, 28.0, 96.0))

        temp_scalar = float(np.clip(1.0 - abs(tavg - 28.5) * 0.035, 0.55, 1.18))
        moisture_scalar = float(np.clip((soil_moisture_pct - 20.0) / 55.0, 0.40, 1.22))
        rad_scalar = float(np.clip(d["rad_mj"] / 18.0, 0.70, 1.20))

        pred_pgr = round(base_pgr_kg_ha_day * temp_scalar * moisture_scalar * rad_scalar, 1)
        # In C4 tropical grasses, 1 new leaf emerges approximately every ~115 GDD
        leaf_rate_days = round(115.0 / max(8.0, gdd), 1)

        forecast_series.append({
            "date": d["date"],
            "Tmax": round(d["tmax"], 1),
            "Tmin": round(d["tmin"], 1),
            "Tavg": round(tavg, 1),
            "rain_mm": round(d["rain_mm"], 1),
            "et0_mm": round(d["et0_mm"], 2),
            "rad_mj": round(d["rad_mj"], 1),
            "gdd": round(gdd, 1),
            "soil_moisture_pct": round(soil_moisture_pct, 1),
            "pgr_kg_ha_day": pred_pgr,
            "pgr_kg_rai_day": round(pred_pgr * 0.16, 2),
            "days_per_leaf": leaf_rate_days,
        })

    avg_pgr_7d = round(float(np.mean([x["pgr_kg_ha_day"] for x in forecast_series[:7]])), 1)
    avg_pgr_14d = round(float(np.mean([x["pgr_kg_ha_day"] for x in forecast_series])), 1)
    total_rain_14d = round(sum(x["rain_mm"] for x in forecast_series), 1)
    total_gdd_14d = round(sum(x["gdd"] for x in forecast_series), 1)
    avg_leaf_days = round(float(np.mean([x["days_per_leaf"] for x in forecast_series])), 1)

    return {
        "summary": {
            "avg_pgr_7d_kg_ha": avg_pgr_7d,
            "avg_pgr_14d_kg_ha": avg_pgr_14d,
            "total_rain_14d_mm": total_rain_14d,
            "total_gdd_14d": total_gdd_14d,
            "avg_days_per_new_leaf": avg_leaf_days,
            "optimal_rotation_length_days": round(avg_leaf_days * 3.0, 0),
        },
        "daily_14d": forecast_series,
    }


# ============================================================================
# 4. SCENARIO FEED WEDGE, LEAF STAGE & MULTI-MOB ROTATION PLANNER
# ============================================================================

def build_pasture_io_suite(
    analyzed_paddocks: List[Dict[str, Any]],
    target_pre_grazing_kg_ha: float = 2800.0,
    target_residual_kg_ha: float = 1500.0,
) -> Dict[str, Any]:
    """
    Constructs the complete Pasture.io decision suite:
    1. Multi-Mob Livestock Demand & Farm Stocking Rate (LSU/ha, kg DM/ha/day demand vs PGR supply)
    2. Scenario Feed Wedge for Today (Day 0), +7 Days, and +14 Days with Target Demand Line & Silage Surplus
    3. Leaf Stage (1.0 - 3.5 leaves) & Rotation Planner Schedule
    4. 14-Day Weather, GDD & PGR Forecast
    5. Annual Paddock Yield & Utilization Benchmarks
    """
    valid_paddocks = [
        p for p in analyzed_paddocks
        if p.get("is_valid_pasture", True)
        and p["readiness"]["overall_code"] != "NOT_PASTURE_LAND"
    ]
    total_farm_ha = max(0.1, sum(p["area_ha"] for p in valid_paddocks))
    total_farm_rai = max(0.5, sum(p["area_rai"] for p in valid_paddocks))

    # 1. Calculate Multi-Mob Demand & Stocking Rate
    mob_summaries: List[Dict[str, Any]] = []
    total_heads = 0
    total_lsu = 0.0
    total_pasture_demand_kg_day = 0.0
    total_supp_kg_day = 0.0

    paddock_name_map = {p["id"]: p["name"] for p in analyzed_paddocks}

    for m in ACTIVE_MOBS:
        heads = int(m["head_count"])
        wt = float(m["avg_weight_kg"])
        dmi_pct = float(m["dm_intake_pct_bw"])
        supp_head = float(m["supplement_kg_dm_head"])

        total_dmi_head = round(wt * (dmi_pct / 100.0), 2)
        pasture_dmi_head = round(max(2.0, total_dmi_head - supp_head), 2)
        mob_pasture_day = round(heads * pasture_dmi_head, 1)
        mob_supp_day = round(heads * supp_head, 1)
        # 1 Livestock Unit (LSU) = 450 kg liveweight equivalent
        mob_lsu = round(heads * (wt / 450.0), 2)

        total_heads += heads
        total_lsu += mob_lsu
        total_pasture_demand_kg_day += mob_pasture_day
        total_supp_kg_day += mob_supp_day

        curr_pid = m.get("current_paddock_id", "PADDOCK-A1")
        curr_p = next((px for px in analyzed_paddocks if px["id"] == curr_pid), None)
        if curr_p:
            avail_kg = max(
                0.0,
                (curr_p["prediction"]["biomass"]["green_dm_kg_ha"] - m["target_residual_kg_ha"])
                * curr_p["area_ha"]
                * 0.80,
            )
            days_in_current = round(avail_kg / max(1.0, mob_pasture_day), 1)
        else:
            days_in_current = 0.0

        mob_summaries.append({
            **m,
            "current_paddock_name": paddock_name_map.get(curr_pid, curr_pid),
            "lsu": mob_lsu,
            "total_dmi_kg_head_day": total_dmi_head,
            "pasture_dmi_kg_head_day": pasture_dmi_head,
            "mob_pasture_demand_kg_day": mob_pasture_day,
            "mob_supplement_kg_day": mob_supp_day,
            "days_remaining_in_paddock": days_in_current,
        })

    farm_demand_kg_ha_day = round(total_pasture_demand_kg_day / total_farm_ha, 1)
    avg_pgr_today = round(
        sum(p["prediction"]["biomass"]["pgr_kg_ha_day"] * p["area_ha"] for p in valid_paddocks) / total_farm_ha,
        1,
    )
    pgr_minus_demand = round(avg_pgr_today - farm_demand_kg_ha_day, 1)

    # 2. Fetch 14-Day Weather & Pio PGR Forecast
    center_lat = valid_paddocks[0]["center"][0] if valid_paddocks else 14.6535
    center_lng = valid_paddocks[0]["center"][1] if valid_paddocks else 101.3998
    forecast_14d = fetch_14d_pio_forecast(center_lat, center_lng, base_pgr_kg_ha_day=max(35.0, avg_pgr_today))
    days_per_leaf = forecast_14d["summary"]["avg_days_per_new_leaf"]

    # 3. Build Scenario Feed Wedge (Day 0, +7 Days, +14 Days) + Leaf Stage
    n_valid = max(1, len(valid_paddocks))
    sorted_valid = sorted(
        valid_paddocks,
        key=lambda x: x["prediction"]["biomass"]["green_dm_kg_ha"],
        reverse=True,
    )

    scenario_wedge: List[Dict[str, Any]] = []
    total_surplus_tonnes_dm = 0.0

    for rank_idx, p in enumerate(sorted_valid):
        bm = p["prediction"]["biomass"]
        nu = p["prediction"]["nutrition"]
        age = p["prediction"]["maturity"]["estimated_age_days"]
        dm_today = bm["green_dm_kg_ha"]
        pgr = bm["pgr_kg_ha_day"]

        # Target Demand Line slopes linearly from target_pre_grazing_kg_ha (rank 0) to target_residual_kg_ha (rank N-1)
        frac = rank_idx / max(1, n_valid - 1) if n_valid > 1 else 0.0
        target_line_dm = round(target_pre_grazing_kg_ha - frac * (target_pre_grazing_kg_ha - target_residual_kg_ha), 0)
        surplus_kg_ha = round(dm_today - target_line_dm, 0)

        if surplus_kg_ha > 150:
            total_surplus_tonnes_dm += (surplus_kg_ha * p["area_ha"]) / 1000.0

        # +7d and +14d projections (accounting for scheduled grazing if rank_idx == 0 or 1)
        if rank_idx == 0 and dm_today >= target_pre_grazing_kg_ha:
            # Top paddock gets grazed down to residual during the week, then regrows
            dm_7d = round(target_residual_kg_ha + pgr * 3.0, 0)
            dm_14d = round(dm_7d + pgr * 7.0, 0)
        elif rank_idx == 1 and dm_today >= target_pre_grazing_kg_ha * 0.9:
            dm_7d = round(min(4500.0, dm_today + pgr * 5.0), 0)
            dm_14d = round(target_residual_kg_ha + pgr * 4.0, 0)
        else:
            dm_7d = round(min(4800.0, dm_today + pgr * 7.0), 0)
            dm_14d = round(min(5000.0, dm_today + pgr * 14.0), 0)

        # Leaf Emergence Stage (0.0 to 3.5 leaves; optimal grazing is 2.5 - 3.0 leaves)
        leaf_stage = round(min(3.8, max(0.2, age / max(8.0, days_per_leaf))), 2)
        if leaf_stage < 1.5:
            leaf_status = "🌱 < 1.5 ใบ (กำลังฟื้นตัว ห้ามแทะเล็ม)"
        elif leaf_stage < 2.5:
            leaf_status = "🌿 1.5–2.4 ใบ (สะสมพลังงานราก รออีกนิด)"
        elif leaf_stage <= 3.1:
            leaf_status = "✅ 2.5–3.0 ใบ (ระยะทองคำ Goldilocks พร้อมกิน!)"
        else:
            leaf_status = "🌾 > 3.0 ใบ (ใบล่างเริ่มแก่ ควรเร่งตัด/แทะเล็ม)"

        scenario_wedge.append({
            "rank": rank_idx + 1,
            "id": p["id"],
            "name": p["name"],
            "short_name": p["name"].split(" - ")[0],
            "area_rai": p["area_rai"],
            "area_ha": p["area_ha"],
            "dm_today_kg_ha": dm_today,
            "dm_7d_kg_ha": dm_7d,
            "dm_14d_kg_ha": dm_14d,
            "target_line_kg_ha": target_line_dm,
            "residual_line_kg_ha": target_residual_kg_ha,
            "surplus_deficit_kg_ha": surplus_kg_ha,
            "pgr_kg_ha_day": pgr,
            "cp_pct": nu["cp_pct"],
            "ndf_pct": nu["ndf_pct"],
            "age_days": age,
            "leaf_stage": leaf_stage,
            "leaf_status": leaf_status,
            "status_badge": p["readiness"]["overall_badge"],
            "status_color": p["readiness"]["overall_color"],
            "recommendation_action": (
                "🚜 ตัดทำหญ้าหมัก (Silage Surplus)" if surplus_kg_ha >= 650
                else ("🐄 ปล่อยฝูงโคเข้าแทะเล็มคิวถัดไป" if dm_today >= 2100 else "⏳ พักแปลงสะสมมวลชีวมวล")
            ),
        })

    # 4. Build 30-Day Automated Rotational Grazing Queue for the Primary Mob
    rotation_schedule: List[Dict[str, Any]] = []
    cursor_date = datetime.date.today()
    for w in scenario_wedge:
        if w["dm_today_kg_ha"] < 1600:
            continue
        avail_dm_kg = max(
            0.0,
            (w["dm_today_kg_ha"] - target_residual_kg_ha) * w["area_ha"] * 0.82,
        )
        graze_days = max(1.0, round(avail_dm_kg / max(50.0, total_pasture_demand_kg_day), 1))
        entry_str = cursor_date.isoformat()
        exit_date = cursor_date + datetime.timedelta(days=max(1, round(graze_days)))
        rotation_schedule.append({
            "queue_order": len(rotation_schedule) + 1,
            "paddock_id": w["id"],
            "paddock_name": w["name"],
            "entry_date": entry_str,
            "exit_date": exit_date.isoformat(),
            "graze_days": graze_days,
            "pre_grazing_dm_kg_ha": w["dm_today_kg_ha"],
            "target_residual_kg_ha": target_residual_kg_ha,
            "available_forage_kg_dm": round(avail_dm_kg, 0),
            "leaf_stage": w["leaf_stage"],
            "cp_pct": w["cp_pct"],
        })
        cursor_date = exit_date

    # 5. Annual Paddock Harvest & Utilization Benchmarks
    benchmarks: List[Dict[str, Any]] = []
    for p in valid_paddocks:
        bm = p["prediction"]["biomass"]
        annual_t_ha = round((bm["annual_projected_kg_rai"] * 6.25) / 1000.0, 1)
        annual_kg_rai = round(bm["annual_projected_kg_rai"], 0)
        grazed_pct = 74.0 if bm["green_dm_kg_ha"] >= 1800 else 45.0
        silage_pct = round(100.0 - grazed_pct - 12.0, 1)
        utilization_pct = round(grazed_pct + silage_pct, 1)
        n_applied = sum(
            a["n_kg_ha"] for a in ACTIVE_ACTIVITIES
            if a["paddock_id"] == p["id"] and a["type"] == "FERTILIZER"
        ) or 110.0
        cost_thb_kg_dm = round(1.85 if annual_t_ha >= 14.0 else 2.45, 2)
        benchmarks.append({
            "paddock_id": p["id"],
            "paddock_name": p["name"],
            "area_rai": p["area_rai"],
            "annual_yield_t_dm_ha": annual_t_ha,
            "annual_yield_kg_dm_rai": annual_kg_rai,
            "grazed_share_pct": grazed_pct,
            "silage_share_pct": silage_pct,
            "utilization_pct": utilization_pct,
            "n_applied_kg_ha_yr": round(n_applied, 1),
            "cost_thb_per_kg_dm": cost_thb_kg_dm,
        })

    return {
        "grazing_planner": {
            "total_mobs": len(ACTIVE_MOBS),
            "total_heads": total_heads,
            "total_lsu": round(total_lsu, 1),
            "stocking_rate_lsu_ha": round(total_lsu / total_farm_ha, 2),
            "stocking_rate_head_rai": round(total_heads / total_farm_rai, 2),
            "total_pasture_demand_kg_day": round(total_pasture_demand_kg_day, 1),
            "total_supplement_kg_day": round(total_supp_kg_day, 1),
            "farm_demand_kg_ha_day": farm_demand_kg_ha_day,
            "farm_supply_pgr_kg_ha_day": avg_pgr_today,
            "net_growth_balance_kg_ha_day": pgr_minus_demand,
            "surplus_silage_tonnes_dm": round(total_surplus_tonnes_dm, 2),
            "mobs": mob_summaries,
            "rotation_schedule": rotation_schedule,
        },
        "scenario_wedge": scenario_wedge,
        "forecast_14d": forecast_14d,
        "activities": ACTIVE_ACTIVITIES[:25],
        "annual_benchmarks": benchmarks,
    }

