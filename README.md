# 🛰️ GrassSense × Pasture.io — Satellite AI Pasture Intelligence System

ระบบประเมินปริมาณมวลหญ้าแห้ง (**DM / Biomass**), โปรตีนหยาบ (**CP**), เยื่อใย (**NDF & ADF**), อายุหญ้า (**Grass Age**) และวางแผนการปล่อยโคหมุนเวียน (**Pasture.io Feed Wedge & Rotational Grazing**) จากข้อมูลดาวเทียม **Sentinel-1 SAR + Sentinel-2 L2A Multispectral** ร่วมกับ **Machine Learning (SVR & Random Forest)**

พัฒนาตามคอนเซปต์ **GrassSense (G14 Albert's Meat & Balls — FRA361/362 Studio IV)** และงานวิจัยอ้างอิง:
1. **Fernandes et al. (2024)**, *Scientific Reports* 14:8704 (`DOI: 10.1038/s41598-024-59160-x`) — *Using Sentinel-2 satellite images and machine learning algorithms to predict tropical pasture forage mass, crude protein, and fiber content*
2. **Komisarenko et al. (2022)**, *Scientific Reports* (`DOI: 10.1038/s41598-022-04932-6`) — การตรวจจับวันตัด/แทะเล็มหญ้า (`Day 0`) จากการเปลี่ยนแปลงของ `NDVI` ร่วมกับค่า `Sentinel-1 SAR Coherence`
3. **Gargiulo et al. (2020)**, *Remote Sensing* (`DOI: 10.3390/rs12193222`) — การติดตามอัตราการฟื้นตัวของหญ้า (`Pasture Growth Rate - PGR`)

---

## 🚀 โครงสร้างโปรเจค
- [`backend/ml_models.py`](backend/ml_models.py) — คำนวณ 15 ดัชนีพืชพรรณ (`NDVI`, `NDVI8A`, `NDRE`, `CCCI`, `EVI`, `LSWI`, `NBR`, `CIgreen`, `CIredge`, `GDVI`, `GLI`, `LChloI`, `OSAVI`, `SR`, `SRredge`), โมเดล `Support Vector Regression (SVR)` และ `Random Forest (RF)`, สมการผสานเรดาร์+แสง (`AGB SAR+Optical Fusion`), ระบบประเมิน 4 ตัวแปรความพร้อมของหญ้า, `TRL-4 Data Quality Gate` และการสร้างข้อความแจ้งเตือน `LINE Official Account`
- [`backend/satellite_engine.py`](backend/satellite_engine.py) — เชื่อมต่อ **Element84 Earth Search STAC API** (`sentinel-2-l2a`), **Microsoft Planetary Computer STAC API** (`sentinel-1-rtc`) และ **Open-Meteo API** (ดึงข้อมูลอากาศย้อนหลัง 30 วันจริงตามพิกัดแปลง), สร้างพิกเซลความละเอียด `10×10 เมตร` สำหรับแสดง Heatmap รายแปลง และข้อมูล Time-Series ย้อนหลัง 90 วัน
- [`app.py`](app.py) — FastAPI Web Server พร้อม REST API (`/api/farm`, `/api/calculate`, `/api/paddock/add`)
- [`static/index.html`](static/index.html), [`static/styles.css`](static/styles.css), [`static/app.js`](static/app.js) — หน้าเว็บ Dashboard แบบ **Pasture.io × GrassSense** พร้อมแผนที่ดาวเทียม GIS, กราฟ **Feed Wedge**, กราฟ **90-Day Regrowth & Day 0 Detection**, ห้องแล็บคำนวณค่าดิบดาวเทียม และจำลอง **LINE Chatbot**
- [`run_calculation_demo.py`](run_calculation_demo.py) — สคริปต์รันดึงข้อมูลดาวเทียมจริงและแสดงผลการคำนวณครบทุกตัวแปรบน Terminal

---

## 💻 วิธีรันใช้งาน

### 1. รันสคริปต์คำนวณและแสดงผลลัพธ์ทาง Terminal
```bash
python run_calculation_demo.py
```

### 2. รันระบบ Web Dashboard (Pasture.io × GrassSense UI)
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
จากนั้นเปิดเบราว์เซอร์ไปที่ `http://localhost:8000`

