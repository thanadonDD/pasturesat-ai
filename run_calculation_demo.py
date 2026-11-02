"""
GrassSense × Pasture.io — Live Satellite & AI Calculation Verification Script
Fetches real Sentinel-2 / Sentinel-1 STAC metadata + Open-Meteo 30-day weather
and computes all 15 Vegetation Indices, DM/Biomass (SVR, RF, SAR-Optical Fusion),
Crude Protein (CP %), Fiber (NDF/ADF %), Grass Rest Age, Pasture.io Feed Wedge,
and LINE OA Notification output.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

from backend.satellite_engine import DEFAULT_PADDOCKS, analyze_paddock_complete
from app import get_farm_overview


def main():
    print("=" * 88)
    print("🛰️  GrassSense × Pasture.io — ระบบดึงข้อมูลดาวเทียม Sentinel-1/2 & คำนวณ NDVI, DM, CP, NDF")
    print("=" * 88)

    farm = get_farm_overview(herd_count=20, cow_weight_kg=350.0)
    s = farm["summary"]

    print(f"\n📌 สรุปภาพรวมฟาร์ม: {farm['farm_name']}")
    print(f"   • พื้นที่รวมทั้งหมด : {s['total_area_rai']} ไร่ ({s['total_area_ha']} ha | {s['total_paddocks']} แปลง)")
    print(f"   • แปลงพร้อมแทะเล็มทันที : {s['ready_paddocks']} / {s['total_paddocks']} แปลง (รองรับฝูงโค {farm['herd_count']} ตัว ได้รวม {s['rotation_grazing_days_ready']} วัน)")
    print(f"   • ค่าเฉลี่ยทั้งฟาร์ม   : NDVI = {s['avg_ndvi']:.3f} | Green DM = {s['avg_green_dm_kg_ha']:,.0f} kg DM/ha | PGR = +{s['avg_pgr_kg_ha_day']} kg/ha/วัน | CP = {s['avg_cp_pct']}% | NDF = {s['avg_ndf_pct']}%")

    print("\n" + "-" * 88)
    print("📊 ตาราง Pasture.io Feed Wedge (เรียงตามปริมาณหญ้าใบเขียวพร้อมกิน Green DM มาก -> น้อย)")
    print("-" * 88)
    header = f"{'แปลงหญ้า':<36} | {'ไร่':>5} | {'NDVI':>5} | {'NDRE':>5} | {'CCCI':>5} | {'Green DM(kg/ha)':>15} | {'DMรวม(กก./ไร่)':>14} | {'CP(%)':>6} | {'NDF(%)':>6} | {'อายุ':>6} | {'สถานะ'}"
    print(header)
    print("-" * len(header))
    for w in farm["feed_wedge"]:
        print(
            f"{w['name'][:36]:<36} | {w['area_rai']:>5.1f} | {w['NDVI']:>5.3f} | {w['NDRE']:>5.3f} | {w['CCCI']:>5.3f} | "
            f"{w['green_dm_kg_ha']:>15,.1f} | {w['total_dm_kg_rai']:>14,.1f} | {w['cp_pct']:>5.1f}% | {w['ndf_pct']:>5.1f}% | "
            f"{w['age_days']:>4.0f}วัน | {w['status_badge']}"
        )

    # Detailed breakdown of Paddock A1 (Ready) & Paddock B2 (Bloat Risk) & Paddock C1 (Old Grass)
    p0 = farm["paddocks"][0]
    pred = p0["prediction"]
    stac = p0["stac_metadata"]
    s2 = stac["selected_s2_scene"]
    s1 = stac["selected_s1_scene"]

    print("\n" + "=" * 88)
    print(f"🔍 เจาะลึกผลการดึงข้อมูลดาวเทียมและการคำนวณรายแปลง: {p0['name']}")
    print("=" * 88)
    print(f"1. ข้อมูลดาวเทียมจริงจาก STAC Catalog ({stac['stac_source']}):")
    print(f"   • Sentinel-2 L2A Scene : {s2['id']} | วันที่ถ่าย: {s2['datetime']} | เมฆปกคลุม: {s2['cloud_cover_pct']}% | Tile: {s2['mgrs_tile']}")
    print(f"   • Sentinel-1 SAR Scene : {s1['id']} | วันที่ถ่าย: {s1['datetime']} | โหมด: {s1['instrument_mode']} ({', '.join(s1['polarizations'])})")
    print(f"   • สภาพอากาศ Open-Meteo : Tmax={pred['weather']['Tmax']}°C, Tmin={pred['weather']['Tmin']}°C, Tavg={pred['weather']['Tavg']}°C, RH={pred['weather']['RH']}%, ฝน 30 วัน={pred['weather']['Rainfall_mm']} มม. ({pred['weather']['ND']} วัน)")

    print("\n2. ค่าการสะท้อนแสง Sentinel-2 ทั้ง 10 ย่านความถี่ (Surface Reflectance) & Sentinel-1 SAR:")
    b_str = " | ".join(f"{k}={v:.4f}" for k, v in pred["bands"].items())
    print(f"   • Spectral Bands : {b_str}")
    print(f"   • SAR Backscatter: VH = {pred['sar']['VH_dB']} dB | VV = {pred['sar']['VV_dB']} dB | VH/VV Ratio = {pred['sar']['VH_VV_ratio_dB']} dB")

    print("\n3. ดัชนีพืชพรรณทั้ง 15 ตัวแปร (Fernandes et al., 2024 Table 6 + GrassSense PDF):")
    vis = pred["vegetation_indices"]
    for i, (k, v) in enumerate(vis.items(), 1):
        end_ch = "\n" if i % 5 == 0 else "   "
        print(f"   [{i:02d}] {k:<8}: {v:>7.4f}", end=end_ch)
    if len(vis) % 5 != 0:
        print()

    bm = pred["biomass"]
    nu = pred["nutrition"]
    print("\n4. ผลการพยากรณ์ด้วย Machine Learning (SVR & Random Forest) + สมการ Multi-Sensor AGB:")
    print(f"   • มวลหญ้าแห้งรวม (Total Dry FM)      : {bm['dry_fm_gm2']} g/m² = {bm['total_dm_kg_ha']:,.1f} kg DM/ha = {bm['total_dm_kg_rai']:,.1f} กก./ไร่ (SVR={bm['dry_fm_svr_gm2']} g/m², RF={bm['dry_fm_rf_gm2']} g/m²)")
    print(f"   • มวลหญ้าใบเขียวพร้อมกิน (Green DM)   : {bm['green_dm_gm2']} g/m² = {bm['green_dm_kg_ha']:,.1f} kg DM/ha = {bm['green_dm_kg_rai']:,.1f} กก./ไร่")
    print(f"   • มวลใบหญ้าล้วน (Dry Leaf FM)        : {bm['leaf_dm_gm2']} g/m² = {bm['leaf_dm_kg_ha']:,.1f} kg DM/ha = {bm['leaf_dm_kg_rai']:,.1f} กก./ไร่")
    print(f"   • สมการผสานเรดาร์+แสง (SAR+Optical)   : {bm['agb_sar_optical_fusion_gm2']} g/m² = {bm['agb_sar_optical_fusion_kg_rai']:,.1f} กก./ไร่ [{bm['agb_equation']}]")
    print(f"   • อัตราการเติบโตของหญ้า (PGR)         : +{bm['pgr_kg_ha_day']} kg DM/ha/วัน (+{bm['pgr_kg_rai_day']} กก./ไร่/วัน) | คาดการณ์ทั้งปี: {bm['annual_projected_kg_rai']:,.0f} กก./ไร่/ปี")
    print(f"   • โปรตีนหยาบ (Crude Protein - CP)     : {nu['cp_pct']}% DM (SVR={nu['cp_svr_pct']}%, RF={nu['cp_rf_pct']}%, ไนโตรเจน %N={nu['nitrogen_pct']}%)")
    print(f"   • เยื่อใยอาหาร (Fiber - NDF & ADF)     : NDF = {nu['ndf_pct']}% DM (SVR={nu['ndf_svr_pct']}%, RF={nu['ndf_rf_pct']}%) | ADF = {nu['adf_pct']}% DM | ย่อยได้ TDN = {nu['digestibility_tdn_pct']}%")

    print("\n5. ตัวอย่างข้อความแจ้งเตือนผ่าน LINE Official Account สำหรับเกษตรกรรายย่อย:")
    print("-" * 60)
    print(p0["readiness"]["line_notification_text"])
    print("-" * 60)

    # Test 6: Verify what happens when a user draws a polygon over the BROWN PLOWED DIRT FIELD in the screenshot!
    print("\n" + "=" * 88)
    print("🟤 ทดสอบการวาดขอบเขตทับแปลงดินเปลือย/ไถพรวนสีน้ำตาล (แปลงในรูปที่ผู้ใช้แคปหน้าจอ พิกัด 14.6530, 101.4002):")
    print("=" * 88)
    brown_paddock = analyze_paddock_complete({
        "id": "TEST-BROWN-SOIL",
        "name": "แปลงทดสอบ (วาดใหม่บนดินสีน้ำตาล)",
        "species": "giant_napier",
        "coordinates": [
            [14.65339, 101.39967],
            [14.65339, 101.40059],
            [14.65248, 101.40059],
            [14.65248, 101.39967],
        ],
    })
    lc_b = brown_paddock["land_cover_stats"]
    print(f"   • จำนวนพิกเซล 10×10m ทั้งหมด      : {lc_b['total_pixels_10m']} พิกเซล")
    print(f"   • พิกเซลดินเปลือย/หญ้าแห้งสีน้ำตาล : {lc_b['bare_soil_pixels_count']} พิกเซล ({lc_b['bare_soil_fraction']*100:.1f}%)")
    print(f"   • พิกเซลต้นไม้/หย่อมสีเขียวในแปลง   : {lc_b['veg_pixels_count']} พิกเซล ({lc_b['veg_fraction']*100:.1f}%)")
    print(f"   • ค่า NDVI / NDRE จริงของแปลง     : NDVI = {brown_paddock['prediction']['vegetation_indices']['NDVI']} | NDRE = {brown_paddock['prediction']['vegetation_indices']['NDRE']}")
    print(f"   • สถานะที่ระบบตัดสิน               : {brown_paddock['readiness']['overall_badge']}")
    print(f"   • มวลหญ้าสดพร้อมกิน (Green DM)    : {brown_paddock['prediction']['biomass']['green_dm_kg_ha']} kg DM/ha ({brown_paddock['prediction']['biomass']['total_dm_kg_rai']} กก./ไร่)")
    print(f"   • อายุหญ้า                         : {brown_paddock['readiness']['variables_check']['age']['value']} ({brown_paddock['readiness']['variables_check']['age']['detail']})")
    print(f"   • คำแนะนำระบบ                      : {brown_paddock['readiness']['action_recommendation']}")

    # Test 7: Verify what happens when a user draws a polygon over a BUILDING / CONCRETE PARKING LOT
    print("\n" + "=" * 88)
    print("🏢 ทดสอบการวาดขอบเขตทับตึก / หลังคา / ลานจอดรถคอนกรีต (พิกัด 14.6698, 101.4038):")
    print("=" * 88)
    bldg_paddock = analyze_paddock_complete({
        "id": "TEST-BUILDING",
        "name": "ทดสอบตีกรอบทับตึกและลานปูน",
        "species": "giant_napier",
        "coordinates": [
            [14.6704, 101.4032],
            [14.6704, 101.4045],
            [14.6692, 101.4045],
            [14.6692, 101.4032],
        ],
    })
    lc = bldg_paddock["land_cover_stats"]
    print(f"   • จำนวนพิกเซล 10×10m ทั้งหมด : {lc['total_pixels_10m']} พิกเซล")
    print(f"   • พิกเซลตึก/คอนกรีต/ไร้พืชพรรณ : {lc['built_pixels_count']} พิกเซล ({lc['built_fraction']*100:.1f}%)")
    print(f"   • พิกเซลพืชพรรณ (ต้นไม้ริมขอบ): {lc['veg_pixels_count']} พิกเซล ({lc['veg_fraction']*100:.1f}%)")
    print(f"   • สถานะที่ระบบตัดสิน          : {bldg_paddock['readiness']['overall_badge']}")
    print(f"   • มวลหญ้าแห้ง (DM)            : {bldg_paddock['prediction']['biomass']['total_dm_kg_rai']} กก./ไร่ ({bldg_paddock['prediction']['biomass']['green_dm_kg_ha']} kg DM/ha)")
    print(f"   • โปรตีน (CP) / เยื่อใย (NDF)  : CP = {bldg_paddock['prediction']['nutrition']['cp_pct']}% | NDF = {bldg_paddock['prediction']['nutrition']['ndf_pct']}%")
    print(f"   • อายุหญ้า                    : {bldg_paddock['readiness']['variables_check']['age']['value']} ({bldg_paddock['readiness']['variables_check']['age']['detail']})")
    print(f"   • คำแนะนำระบบ                 : {bldg_paddock['readiness']['action_recommendation']}")


if __name__ == "__main__":
    main()

