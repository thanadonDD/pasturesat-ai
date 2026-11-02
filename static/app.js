/**
 * GrassSense × Pasture.io — Frontend Interactive Controller
 */

let mapInstance = null;
let paddockLayersGroup = null;
let heatmapCellsGroup = null;
let drawTempMarkers = [];
let drawTempLine = null;
let isDrawingMode = false;
let drawnCoords = [];

let farmData = null;
let selectedPaddockId = null;
let currentHeatmapMetric = "NDVI";
let showHeatmapGrid = true;

let feedWedgeChartInstance = null;
let timeseriesChartInstance = null;
let scenarioWedgeChartInstance = null;
let pioForecast14dChartInstance = null;
let gtComparisonChartInstance = null;
let gtProofMapInstance = null;
let gtProofMarkersGroup = null;
let gtClickedMarker = null;
let currentScenarioHorizon = 0;
let currentGtChartMetric = "DM";
let currentGtSourceFilter = "ALL";

// Initialize Map & Load Farm Data on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initSatelliteMap();
  loadFarmOverview();
  runCustomSpectralCalc();

  document.getElementById("refreshFarmBtn").addEventListener("click", () => {
    loadFarmOverview();
  });

  document.getElementById("heatmapLayerSelect").addEventListener("change", (e) => {
    currentHeatmapMetric = e.target.value;
    updateLegendForMetric(currentHeatmapMetric);
    renderMapPolygonsAndHeatmap();
  });

  document.getElementById("toggleHeatmapBtn").addEventListener("click", () => {
    showHeatmapGrid = !showHeatmapGrid;
    const btn = document.getElementById("toggleHeatmapBtn");
    btn.textContent = showHeatmapGrid ? "👁️ ซ่อนสีพิกเซล (ดูภาพจริง)" : "🎨 แสดงสีพิกเซล 10×10m";
    renderMapPolygonsAndHeatmap();
  });

  document.getElementById("drawPaddockBtn").addEventListener("click", startDrawMode);
  document.getElementById("undoDrawPtBtn").addEventListener("click", undoLastDrawPoint);
  document.getElementById("clearDrawPtsBtn").addEventListener("click", clearAllDrawPoints);
  document.getElementById("cancelDrawBtn").addEventListener("click", cancelDrawMode);
  document.getElementById("finishDrawBtn").addEventListener("click", submitDrawnPaddock);
  document.getElementById("deleteSelectedBtn").addEventListener("click", () => {
    if (selectedPaddockId) deletePaddockById(selectedPaddockId);
  });
  document.getElementById("resetPaddocksBtn").addEventListener("click", resetAllPaddocks);
  document.getElementById("runCustomCalcBtn").addEventListener("click", runCustomSpectralCalc);
  const submitActBtn = document.getElementById("submitActivityBtn");
  if (submitActBtn) {
    submitActBtn.addEventListener("click", submitPaddockActivity);
  }
  const runGtLabBtn = document.getElementById("runGtLabCheckBtn");
  if (runGtLabBtn) {
    runGtLabBtn.addEventListener("click", runCustomGtLabCheck);
  }

  document.querySelectorAll(".main-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      if (targetTab) switchMainTab(targetTab);
    });
  });

  document.getElementById("copyLineBtn").addEventListener("click", () => {
    const txt = document.getElementById("linePreviewText").textContent;
    navigator.clipboard.writeText(txt);
    const btn = document.getElementById("copyLineBtn");
    const old = btn.textContent;
    btn.textContent = "✅ คัดลอกข้อความเข้าคลิปบอร์ดแล้ว!";
    setTimeout(() => (btn.textContent = old), 2000);
  });
});

function switchMainTab(tabId) {
  document.querySelectorAll(".tab-view").forEach((el) => {
    el.classList.toggle("active", el.id === tabId);
  });
  document.querySelectorAll(".main-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
  });
  if (tabId === "tab-overview" && mapInstance) {
    setTimeout(() => mapInstance.invalidateSize(), 120);
  }
  if (tabId === "tab-ground-truth" && gtProofMapInstance) {
    setTimeout(() => gtProofMapInstance.invalidateSize(), 120);
  }
}
window.switchMainTab = switchMainTab;
window.loadPreset = loadPreset;
window.deletePaddockById = deletePaddockById;

function initSatelliteMap() {
  mapInstance = L.map("satelliteMap", {
    center: [14.6535, 101.3998],
    zoom: 16,
  });

  // High-res Esri World Imagery Satellite Tiles
  const esriSat = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Tiles &copy; Esri — Source: Esri, Copernicus Sentinel-2, Maxar",
      maxZoom: 19,
    }
  );
  esriSat.addTo(mapInstance);

  heatmapCellsGroup = L.layerGroup().addTo(mapInstance);
  paddockLayersGroup = L.layerGroup().addTo(mapInstance);

  mapInstance.on("click", (e) => {
    if (!isDrawingMode) return;
    const pt = [e.latlng.lat, e.latlng.lng];
    drawnCoords.push(pt);
    const m = L.circleMarker(pt, {
      radius: 5,
      color: "#22d3ee",
      fillColor: "#06b6d4",
      fillOpacity: 1,
    }).addTo(mapInstance);
    drawTempMarkers.push(m);
    updateDrawPreviewLine();
  });
}

function updateDrawPreviewLine() {
  const counterEl = document.getElementById("drawPointCounter");
  if (counterEl) {
    counterEl.textContent = `คลิกแล้ว ${drawnCoords.length} จุด`;
  }
  if (drawTempLine) {
    mapInstance.removeLayer(drawTempLine);
    drawTempLine = null;
  }
  if (drawnCoords.length >= 2) {
    drawTempLine = L.polygon(drawnCoords, {
      color: "#22d3ee",
      weight: 2.5,
      dashArray: "5,5",
      fillOpacity: 0.22,
    }).addTo(mapInstance);
  }
}

async function loadFarmOverview() {
  const herdCount = parseInt(document.getElementById("herdCountInput").value || "20", 10);
  const cowWeight = parseFloat(document.getElementById("cowWeightInput").value || "350");

  const resp = await fetch(`/api/farm?herd_count=${herdCount}&cow_weight_kg=${cowWeight}`);
  farmData = await resp.json();

  if (!selectedPaddockId && farmData.paddocks.length > 0) {
    selectedPaddockId = farmData.paddocks[0].id;
  }

  renderKpiSummary(farmData.summary);
  renderQuickPaddockPills();
  renderMapPolygonsAndHeatmap();
  renderFeedWedgeChart(farmData.feed_wedge);
  renderPaddockTable(farmData.feed_wedge);
  if (farmData.pasture_io_suite) {
    renderPastureIoSuite(farmData.pasture_io_suite);
  }
  if (farmData.ground_truth_validation) {
    renderGroundTruthValidationTab(farmData.ground_truth_validation);
  }

  const selectedObj =
    farmData.paddocks.find((p) => p.id === selectedPaddockId) || farmData.paddocks[0];
  if (selectedObj) {
    renderSelectedPaddockDetails(selectedObj);
  }
}

function renderQuickPaddockPills() {
  if (!farmData || !farmData.paddocks) return;
  const containers = [
    document.getElementById("quickPaddockPills"),
    document.getElementById("quickPaddockPillsRaw"),
  ];
  containers.forEach((box) => {
    if (!box) return;
    box.innerHTML = "";
    farmData.paddocks.forEach((p) => {
      const btn = document.createElement("button");
      btn.className = "paddock-pill" + (p.id === selectedPaddockId ? " active" : "");
      const shortName = p.name.split(" - ")[0];
      const dotColor = (p.readiness && p.readiness.overall_color) || "#10b981";
      btn.innerHTML = `<span class="paddock-pill-dot" style="background:${dotColor}"></span><span>${shortName}</span>`;
      btn.onclick = () => selectPaddock(p.id, true);
      box.appendChild(btn);
    });
  });
}

function renderKpiSummary(sum) {
  document.getElementById("kpiArea").textContent = `${sum.total_area_rai.toFixed(1)} ไร่`;
  document.getElementById("kpiAreaHa").textContent = `${sum.total_area_ha.toFixed(2)} ha (${sum.total_paddocks} แปลง)`;
  document.getElementById("kpiReadyCount").textContent = `${sum.ready_paddocks} / ${sum.total_paddocks} แปลง`;
  document.getElementById("kpiRotationDays").textContent = `รองรับฝูงโคได้รวม ${sum.rotation_grazing_days_ready.toFixed(1)} วัน`;
  document.getElementById("kpiAvgDm").textContent = `${sum.avg_green_dm_kg_ha.toLocaleString()} kg DM/ha`;
  document.getElementById("kpiAvgPgr").textContent = `อัตราโตเฉลี่ย (PGR): +${sum.avg_pgr_kg_ha_day} kg DM/ha/วัน`;
  document.getElementById("kpiAvgCp").textContent = `${sum.avg_cp_pct.toFixed(1)}%`;
  document.getElementById("kpiAvgNdf").textContent = `${sum.avg_ndf_pct.toFixed(1)}%`;
  document.getElementById("kpiAvgNdvi").textContent = sum.avg_ndvi.toFixed(3);
}

// Color interpolation for spatial heatmap cells
function getMetricColor(metric, val) {
  let norm = 0.5;
  if (metric === "NDVI") {
    norm = (val - 0.25) / (0.85 - 0.25);
  } else if (metric === "NDRE") {
    norm = (val - 0.15) / (0.62 - 0.15);
  } else if (metric === "CCCI") {
    norm = (val - 0.45) / (0.95 - 0.45);
  } else if (metric === "DM_kg_ha") {
    norm = (val - 900) / (3500 - 900);
  } else if (metric === "CP_pct") {
    norm = (val - 6.0) / (17.0 - 6.0);
  } else if (metric === "NDF_pct") {
    // Lower/moderate NDF is greener, high NDF (>70%) is orange/red
    norm = 1.0 - (val - 52.0) / (74.0 - 52.0);
  }
  norm = Math.max(0, Math.min(1, norm));

  if (norm < 0.25) return "#ef4444";
  if (norm < 0.45) return "#f59e0b";
  if (norm < 0.65) return "#eab308";
  if (norm < 0.82) return "#84cc16";
  return "#10b981";
}

function updateLegendForMetric(metric) {
  const titleEl = document.getElementById("legendTitle");
  const minEl = document.getElementById("legendMin");
  const midEl = document.getElementById("legendMid");
  const maxEl = document.getElementById("legendMax");

  const configs = {
    NDVI: ["ระดับค่า NDVI (10m Pixel):", "ต่ำ (0.25)", "ปานกลาง (0.55)", "สมบูรณ์สูง (0.85)"],
    NDRE: ["ระดับค่า NDRE (Red-Edge ไนโตรเจน):", "ต่ำ (0.15)", "ปานกลาง (0.38)", "คลอโรฟิลล์สูง (0.62)"],
    CCCI: ["ระดับค่า CCCI (Canopy Chlorophyll):", "ต่ำ (0.45)", "สมดุล (0.70)", "เข้มข้นสูง (0.95)"],
    DM_kg_ha: ["มวลหญ้าพร้อมกิน Green DM (kg/ha):", "น้อย (<1,200)", "ปานกลาง (2,200)", "พร้อมตัด (3,200+)"],
    CP_pct: ["โปรตีนหยาบ Crude Protein (%DM):", "หญ้าแก่ (<7%)", "เหมาะสม (11-13%)", "โปรตีนสูง (16%+)"],
    NDF_pct: ["เยื่อใย NDF (%DM - เขียว=ย่อยง่าย):", "สูง/แก่ (>71%)", "เหมาะสม (64-68%)", "ย่อยง่าย (<58%)"],
  };
  const c = configs[metric] || configs.NDVI;
  titleEl.textContent = c[0];
  minEl.textContent = c[1];
  midEl.textContent = c[2];
  maxEl.textContent = c[3];
}

function renderMapPolygonsAndHeatmap() {
  if (!farmData) return;
  paddockLayersGroup.clearLayers();
  heatmapCellsGroup.clearLayers();

  farmData.paddocks.forEach((p) => {
    const isSelected = p.id === selectedPaddockId;

    // 1. Render REAL 10m x 10m Sentinel-2 Spatial Heatmap Grid Cells inside the paddock
    if (showHeatmapGrid) {
      (p.spatial_grid || []).forEach((cell) => {
        let fillCol = "#64748b";
        if (cell.is_vegetation === false) {
          if (cell.land_class === "WATER") fillCol = "#0284c7";
          else if (cell.land_class === "BARE_SOIL") fillCol = "#ef4444";
          else fillCol = "#64748b"; // Building / Road / Concrete in distinct Slate Grey
        } else {
          const val = cell[currentHeatmapMetric];
          fillCol = getMetricColor(currentHeatmapMetric, val);
        }
        const rect = L.rectangle(cell.bounds, {
          stroke: true,
          color: "rgba(15,23,42,0.22)",
          weight: 0.4,
          fillColor: fillCol,
          fillOpacity: isSelected ? 0.65 : 0.45,
          interactive: false,
        });
        heatmapCellsGroup.addLayer(rect);
      });
    }

    // 2. Render Paddock Boundary Polygon
    const poly = L.polygon(p.coordinates, {
      color: isSelected ? "#ffffff" : p.readiness.overall_color,
      weight: isSelected ? 3.5 : 2,
      fillColor: p.readiness.overall_color,
      fillOpacity: 0.08,
    });

    const isPasture = p.is_valid_pasture !== false;
    const lc = p.land_cover_stats || {};
    const isBareSoilPaddock = p.readiness && p.readiness.overall_code === "REST_BARE_OR_DRY_SOIL";
    let tooltipHtml = "";
    if (!isPasture) {
      tooltipHtml = `<div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:12px;color:#fca5a5;">
          <b>🚫 ${p.name} — ไม่ใช่พื้นที่แปลงหญ้า!</b><br/>
          • ตรวจพบตึก/คอนกรีต/พื้นที่ไร้พืชพรรณ: <b>${((lc.built_fraction || 0) * 100).toFixed(0)}%</b> ของแปลง<br/>
          • NDVI เฉลี่ย: <b>${p.prediction.vegetation_indices.NDVI}</b> | Red (B4): <b>${p.prediction.bands.B4}</b><br/>
          • มวลหญ้า (DM): <b>0 กก./ไร่</b> | อายุหญ้า: <b>ไม่ใช่แปลงหญ้า</b>
        </div>`;
    } else if (isBareSoilPaddock) {
      tooltipHtml = `<div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:12px;color:#fca5a5;">
          <b>🔴 ${p.name} (${p.area_rai} ไร่) — ยังไม่พร้อมกิน!</b><br/>
          • สภาพพื้นผิว: <b>ดินเปลือย/ไถพรวน/หญ้าแห้งสีน้ำตาล ${((lc.bare_soil_fraction || 0) * 100).toFixed(0)}%</b> (หย่อมสีเขียว ${((lc.veg_fraction || 0) * 100).toFixed(0)}%)<br/>
          • Green DM: <b>${p.prediction.biomass.green_dm_kg_ha.toLocaleString()} kg/ha</b> (${p.prediction.biomass.total_dm_kg_rai.toLocaleString()} กก./ไร่)<br/>
          • อายุหญ้าสด: <b>${p.prediction.maturity.estimated_age_days} วัน (ยังไม่งอก/เพิ่งไถ)</b> | ปล่อยแทะเล็มได้: <b>0 วัน</b><br/>
          • NDVI จริง: <b>${p.prediction.vegetation_indices.NDVI} (สีน้ำตาล)</b> | NDRE: <b>${p.prediction.vegetation_indices.NDRE}</b>
        </div>`;
    } else if (p.readiness && p.readiness.overall_code === "MIXED_LAND_WARNING") {
      tooltipHtml = `<div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:12px;color:#fde68a;">
          <b>⚠️ ${p.name} (${p.area_rai} ไร่) — กรอบคร่อมพื้นที่ผสม!</b><br/>
          • สัดส่วนพื้นผิว: <b>🟢 เขียว ${((lc.veg_fraction || 0) * 100).toFixed(0)}%</b> | <b>🟤 ดินเปลือย ${((lc.bare_soil_fraction || 0) * 100).toFixed(0)}%</b> | <b>🛣️ ถนน/ตึก ${((lc.built_fraction || 0) * 100).toFixed(0)}%</b><br/>
          • คำแนะนำ: <b>กดปุ่ม 🗑️ ลบแปลงนี้ แล้ววาดแยกเฉพาะฝั่งสีเขียว</b>
        </div>`;
    } else {
      tooltipHtml = `<div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:12px;">
          <b>${p.name}</b> (${p.area_rai} ไร่ | พิกเซลพืชสีเขียว ${(lc.veg_fraction * 100 || 100).toFixed(0)}%)<br/>
          • สถานะ: <b style="color:${p.readiness.overall_color};">${p.readiness.overall_badge}</b><br/>
          • Green DM: <b>${p.prediction.biomass.green_dm_kg_ha.toLocaleString()} kg/ha</b> (${p.prediction.biomass.total_dm_kg_rai.toLocaleString()} กก./ไร่)<br/>
          • CP: <b>${p.prediction.nutrition.cp_pct}%</b> | NDF: <b>${p.prediction.nutrition.ndf_pct}%</b> | อายุหญ้า: <b>${p.prediction.maturity.estimated_age_days} วัน</b><br/>
          • NDVI จริง: <b>${p.prediction.vegetation_indices.NDVI}</b> | NDRE: <b>${p.prediction.vegetation_indices.NDRE}</b>
        </div>`;
    }

    poly.bindTooltip(tooltipHtml, { sticky: true });

    poly.on("click", () => {
      if (isDrawingMode) return;
      selectPaddock(p.id);
    });

    paddockLayersGroup.addLayer(poly);
  });
}

function selectPaddock(paddockId, flyTo = false) {
  selectedPaddockId = paddockId;
  renderQuickPaddockPills();
  renderMapPolygonsAndHeatmap();
  renderPaddockTable(farmData.feed_wedge);
  const p = farmData.paddocks.find((x) => x.id === paddockId);
  if (p) {
    renderSelectedPaddockDetails(p);
    if (flyTo && mapInstance && p.coordinates && p.coordinates.length >= 3) {
      mapInstance.flyToBounds(p.coordinates, { maxZoom: 17, duration: 0.55 });
    }
  }
}

function renderFeedWedgeChart(wedgeList) {
  const ctx = document.getElementById("feedWedgeChart").getContext("2d");
  if (feedWedgeChartInstance) {
    feedWedgeChartInstance.destroy();
  }

  const labels = wedgeList.map((w) => w.name.split(" - ")[0]);
  const dmValues = wedgeList.map((w) => w.green_dm_kg_ha);
  const barColors = wedgeList.map((w) => w.status_color);
  const preGrazingTarget = wedgeList.map(() => 2800);
  const postGrazingResidual = wedgeList.map(() => 1500);

  feedWedgeChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "line",
          label: "Target Pre-Grazing (2,800 kg DM/ha)",
          data: preGrazingTarget,
          borderColor: "#34d399",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          type: "line",
          label: "Post-Grazing Residual ตอหญ้าขั้นต่ำ (1,500 kg DM/ha)",
          data: postGrazingResidual,
          borderColor: "#f87171",
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          type: "bar",
          label: "มวลหญ้าพร้อมกินรายแปลง (kg DM/ha)",
          data: dmValues,
          backgroundColor: barColors,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#cbd5e1", font: { size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.08)" },
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.08)" },
          title: { display: true, text: "kg DM / ha", color: "#94a3b8" },
        },
      },
    },
  });
}

function renderPaddockTable(wedgeList) {
  const tbody = document.getElementById("paddockTableBody");
  tbody.innerHTML = "";

  wedgeList.forEach((w) => {
    const tr = document.createElement("tr");
    if (w.id === selectedPaddockId) tr.classList.add("active-row");
    tr.onclick = () => selectPaddock(w.id);

    const isInvalid = w.status_code === "NOT_PASTURE_LAND";
    tr.innerHTML = `
      <td><b>${w.name}</b></td>
      <td>${w.area_rai} ไร่<br/><span style="color:#64748b;font-size:10.5px">${w.area_ha} ha</span></td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:11.2px">
        <span style="color:#a3e635">${w.NDVI.toFixed(2)}</span> /
        <span style="color:#38bdf8">${w.NDRE.toFixed(2)}</span> /
        <span style="color:#f472b6">${w.CCCI.toFixed(2)}</span>
      </td>
      <td><b>${isInvalid ? "0 (ไม่ใช่หญ้า)" : w.green_dm_kg_ha.toLocaleString()}</b></td>
      <td>${isInvalid ? "0" : w.total_dm_kg_rai.toLocaleString()}</td>
      <td style="color:#34d399">${isInvalid ? "0" : "+" + w.pgr_kg_ha_day}</td>
      <td><b>${isInvalid ? "N/A" : w.cp_pct.toFixed(1) + "%"}</b></td>
      <td>${isInvalid ? "N/A" : w.ndf_pct.toFixed(1) + "%"}</td>
      <td>${isInvalid ? "ไม่ใช่แปลงหญ้า" : w.age_days + " วัน"}</td>
      <td><b>${isInvalid ? "0 วัน" : w.grazing_days + " วัน"}</b></td>
      <td>
        <span style="background:${w.status_color};color:#fff;padding:3px 8px;border-radius:99px;font-size:10.8px;font-weight:600;white-space:nowrap;">
          ${w.status_badge.split(" (")[0]}
        </span>
      </td>
      <td>
        <button class="btn-table-del" title="ลบกรอบแปลงนี้ออกจากแผนที่" onclick="event.stopPropagation(); deletePaddockById('${w.id}')">🗑️ ลบ</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderSelectedPaddockDetails(p) {
  const pred = p.prediction;
  const bm = pred.biomass;
  const nutri = pred.nutrition;
  const mat = pred.maturity;
  const vis = pred.vegetation_indices;
  const ready = p.readiness;
  const vc = ready.variables_check;
  const trl = ready.data_quality_trl4;
  const s2 = p.stac_metadata.selected_s2_scene;
  const lc = p.land_cover_stats || {};

  document.getElementById("selectedPaddockTitle").textContent = `🔍 ${p.name}`;
  document.getElementById("selectedPaddockMeta").textContent =
    `${pred.species_name_th} • พื้นที่ ${p.area_rai} ไร่ (${p.area_ha} ha / ${p.valid_pixels_10m} พิกเซลดาวเทียม) • พิกัด [${p.center[0]}, ${p.center[1]}]`;

  const badgeEl = document.getElementById("selectedStatusBadge");
  badgeEl.textContent = ready.overall_badge;
  badgeEl.style.background = ready.overall_color;

  const recBox = document.getElementById("selectedRecommendationBox");
  recBox.style.borderLeftColor = ready.overall_color;
  recBox.innerHTML = `<b>💡 คำแนะนำการจัดการแปลง (Decision Engine):</b> ${ready.action_recommendation}`;

  // Update Land-Cover Breakdown Bar
  const vegPct = Math.round((lc.veg_fraction || 0) * 100);
  const barePct = Math.round((lc.bare_soil_fraction || 0) * 100);
  const builtPct = Math.max(0, 100 - vegPct - barePct);
  document.getElementById("lcTotalPixelsLabel").textContent = `รวม ${lc.total_pixels_10m || p.valid_pixels_10m} พิกเซล`;
  document.getElementById("lcBarVeg").style.width = `${vegPct}%`;
  document.getElementById("lcBarBare").style.width = `${barePct}%`;
  document.getElementById("lcBarBuilt").style.width = `${builtPct}%`;
  document.getElementById("lcPctVeg").textContent = `${vegPct}% (${lc.veg_pixels_count || 0} พิกเซล)`;
  document.getElementById("lcPctBare").textContent = `${barePct}% (${lc.bare_soil_pixels_count || 0} พิกเซล)`;
  document.getElementById("lcPctBuilt").textContent = `${builtPct}% (${lc.built_pixels_count || 0} พิกเซล)`;

  // 1. DM Card
  document.getElementById("varBadgeDm").textContent = vc.dm.label;
  document.getElementById("varValDm").textContent = `${bm.total_dm_kg_rai.toLocaleString()} กก./ไร่ (${bm.green_dm_kg_ha.toLocaleString()} kg/ha)`;
  document.getElementById("varSubDm").textContent =
    `SVR: ${bm.dry_fm_svr_gm2} g/m² | RF: ${bm.dry_fm_rf_gm2} g/m² | SAR+Optic AGB: ${bm.agb_sar_optical_fusion_gm2} g/m²`;
  document.getElementById("varDescDm").textContent = vc.dm.detail;

  // 2. CP Card
  document.getElementById("varBadgeCp").textContent = vc.cp.label;
  document.getElementById("varValCp").textContent = `${nutri.cp_pct}% DM`;
  document.getElementById("varSubCp").textContent =
    `SVR: ${nutri.cp_svr_pct}% | RF: ${nutri.cp_rf_pct}% | ไนโตรเจน (%N): ${nutri.nitrogen_pct}%`;
  document.getElementById("varDescCp").textContent = vc.cp.detail;

  // 3. NDF Card
  document.getElementById("varBadgeNdf").textContent = vc.ndf.label;
  document.getElementById("varValNdf").textContent = `NDF ${nutri.ndf_pct}% | ADF ${nutri.adf_pct}%`;
  document.getElementById("varSubNdf").textContent =
    `SVR: ${nutri.ndf_svr_pct}% | RF: ${nutri.ndf_rf_pct}% | ย่อยได้ TDN: ${nutri.digestibility_tdn_pct}% | ME: ${nutri.metabolizable_energy_mj_kg} MJ/kg`;
  document.getElementById("varDescNdf").textContent = vc.ndf.detail;

  // 4. Age Card
  document.getElementById("varBadgeAge").textContent = vc.age.label;
  if (p.is_valid_pasture === false) {
    document.getElementById("varValAge").textContent = "ไม่ใช่พื้นที่แปลงหญ้า (N/A)";
    document.getElementById("varSubAge").textContent =
      `ตรวจพบพิกเซลตึก/คอนกรีต/ดินเปลือย ${(p.land_cover_stats.built_fraction * 100).toFixed(0)}% (${p.land_cover_stats.built_pixels_count}/${p.land_cover_stats.total_pixels_10m} พิกเซล)`;
  } else {
    document.getElementById("varValAge").textContent = `${mat.estimated_age_days} วัน (+${bm.pgr_kg_rai_day} กก./ไร่/วัน)`;
    document.getElementById("varSubAge").textContent =
      `วันตัด/แทะเล็มล่าสุด (Day 0): ${p.last_cut_date} | PGR: +${bm.pgr_kg_ha_day} kg DM/ha/วัน`;
  }
  document.getElementById("varDescAge").textContent = vc.age.detail;

  // Render 90-day Time-Series Chart
  renderTimeseriesChart(p.timeseries_90d);

  // Render 15 Vegetation Indices + SAR Badges
  const viGrid = document.getElementById("viBadgesGrid");
  if (viGrid) {
    viGrid.innerHTML = "";
    Object.entries(vis).forEach(([k, v]) => {
      const div = document.createElement("div");
      div.className = "vi-item";
      div.innerHTML = `<div class="vi-name">${k}</div><div class="vi-val">${v}</div>`;
      viGrid.appendChild(div);
    });
    ["VH_dB", "VV_dB"].forEach((sk) => {
      const div = document.createElement("div");
      div.className = "vi-item";
      div.innerHTML = `<div class="vi-name">SAR ${sk}</div><div class="vi-val" style="color:#38bdf8">${pred.sar[sk]}</div>`;
      viGrid.appendChild(div);
    });
  }

  // Render Tab 2: Realtime Satellite Provenance & 6-Pass Raw Inspector
  renderRawSatelliteProvenanceTab(p);

  // LINE OA Preview & SDG Box
  document.getElementById("linePreviewText").textContent = ready.line_notification_text;

  const sdg = ready.sdg_and_economics;
  const gp = ready.grazing_plan;
  document.getElementById("sdgImpactBox").innerHTML = `
    <div style="font-weight:700;color:#34d399;margin-bottom:4px;">🌱 ผลลัพธ์ด้านเศรษฐกิจและความยั่งยืน (SDGs 2.3, 8.2, 12.2, 13.2, 15.3):</div>
    <div>• <b>มูลค่าหญ้าพร้อมกินในแปลงนี้:</b> ~${sdg.forage_value_thb.toLocaleString()} บาท (หญ้าพร้อมกินสุทธิ ${gp.total_available_dm_kg.toLocaleString()} กก. DM สำหรับโค ${gp.herd_count} ตัว นาน ${gp.grazing_days_available} วัน)</div>
    <div>• <b>ลดต้นทุนซื้อฟางหน้าแล้ง:</b> ลดลง ~${sdg.dry_season_straw_saving_pct}% (ประหยัด ~${sdg.dry_season_straw_saving_thb.toLocaleString()} บาท/ปี) และลดอาหารข้นเสริม ~${sdg.monthly_supplement_saving_thb.toLocaleString()} บาท/เดือน</div>
    <div>• <b>ลดก๊าซเรือนกระจก:</b> ป้องกันดินอัดแน่นที่ปล่อย N₂O จากปัสสาวะโคเพิ่มขึ้น ${sdg.n2o_compaction_multiplier_avoided} เท่า และลดมีเทน (CH₄) รวมลดคาร์บอน <b>${sdg.co2e_mitigated_ton_yr} tCO₂e/ปี</b></div>
  `;
}

function renderRawSatelliteProvenanceTab(p) {
  const pred = p.prediction;
  const stac = p.stac_metadata;
  const s2 = stac.selected_s2_scene;
  const s1 = stac.selected_s1_scene;
  const trl = p.readiness.data_quality_trl4;
  const w = pred.weather;
  const lc = p.land_cover_stats || {};

  document.getElementById("rawTabPaddockTitle").textContent =
    `🛰️ ที่มาข้อมูลดาวเทียม Realtime & ค่าดิบจากเซนเซอร์: ${p.name}`;
  document.getElementById("rawTabSceneBadge").textContent =
    `Scene ล่าสุด: ${s2.id} (${s2.datetime} UTC)`;

  // 1. Source Cards
  document.getElementById("srcDetailS2").innerHTML =
    `• <b>Scene ID:</b> ${s2.id}<br/>` +
    `• <b>เวลาถ่ายภาพ:</b> ${s2.datetime} UTC (Tile ${s2.mgrs_tile}, EPSG:${s2.epsg || 32647})<br/>` +
    `• <b>เมฆปกคลุม:</b> ${s2.cloud_cover_pct}% | พิกเซลในแปลง: ${lc.total_pixels_10m || p.valid_pixels_10m} พิกเซล`;

  const samplePx = (p.spatial_grid && p.spatial_grid[0]) || { esri_rgb: [65, 95, 55], esri_vari: 0.18 };
  const rgb = samplePx.esri_rgb || [65, 95, 55];
  document.getElementById("srcDetailEsri").innerHTML =
    `• <b>ความละเอียด:</b> 1.19 เมตร/พิกเซล (Zoom Level 17 Tile)<br/>` +
    `• <b>ตัวอย่างสีพื้นผิว RGB:</b> <span class="rgb-swatch" style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>(${rgb[0]}, ${rgb[1]}, ${rgb[2]}) | VARI: ${samplePx.esri_vari}<br/>` +
    `• <b>สัดส่วน:</b> เขียว ${Math.round((lc.veg_fraction || 0) * 100)}% / ดินเปลือย ${Math.round((lc.bare_soil_fraction || 0) * 100)}% / ตึก-ถนน ${Math.round((lc.built_fraction || 0) * 100)}%`;

  document.getElementById("srcDetailS1").innerHTML =
    `• <b>Scene ID:</b> ${s1.id}<br/>` +
    `• <b>เวลาถ่ายภาพ:</b> ${s1.datetime} UTC (${s1.instrument_mode})<br/>` +
    `• <b>ค่าสะท้อนเรดาร์:</b> VH = ${pred.sar.VH_dB} dB | VV = ${pred.sar.VV_dB} dB`;

  document.getElementById("srcDetailWeather").innerHTML =
    `• <b>อุณหภูมิ 30 วัน:</b> เฉลี่ย ${w.Tavg}°C (สูงสุด ${w.Tmax}°C / ต่ำสุด ${w.Tmin}°C)<br/>` +
    `• <b>ความชื้นสัมพัทธ์ (RH):</b> ${w.RH}%<br/>` +
    `• <b>ปริมาณฝนสะสม 30 วัน:</b> ${w.Rainfall_mm} มม. (${w.ND} วันฝนตก)`;

  // 2. Visual 6-Pass Cards Strip + Table 1: 6 Latest Satellite Passes (Reversed so newest pass is at top)
  const passes = [...(p.timeseries_90d || [])].reverse();
  const sixStrip = document.getElementById("sixPassesCardsStrip");
  if (sixStrip) {
    sixStrip.innerHTML = "";
    passes.forEach((row, idx) => {
      const isLatest = idx === 0;
      const ndviPct = Math.max(8, Math.min(100, Math.round(row.NDVI * 100)));
      const card = document.createElement("div");
      card.className = "pass-mini-card" + (isLatest ? " latest-pass" : "");
      card.innerHTML = `
        <div class="pass-mini-top">
          <span class="pass-mini-rank" style="background:${isLatest ? "rgba(16,185,129,0.22);color:#34d399;border:1px solid rgba(16,185,129,0.45)" : "rgba(51,65,85,0.65);color:#cbd5e1"}">
            ${isLatest ? "🟢 ล่าสุด (#1)" : "รอบ #" + (idx + 1)}
          </span>
          <span class="pass-mini-date">${(row.datetime || row.date).slice(0, 10)}</span>
        </div>
        <div class="pass-mini-scene">${row.scene_id} • เมฆ ${row.scene_cloud_pct ?? 11.4}%</div>
        <div class="pass-mini-ndvi-bar">
          <div class="pass-mini-ndvi-fill" style="width:${ndviPct}%"></div>
        </div>
        <div class="pass-mini-metrics">
          <div>NDVI: <b style="color:#a3e635">${row.NDVI.toFixed(3)}</b></div>
          <div>NDRE: <b style="color:#38bdf8">${row.NDRE.toFixed(3)}</b></div>
          <div>B8: <b>${(row.B8 ?? pred.bands.B8).toFixed(3)}</b></div>
          <div>DM: <b style="color:#34d399">${row.DM_kg_ha.toLocaleString()}</b></div>
        </div>
      `;
      sixStrip.appendChild(card);
    });
  }

  const sixBody = document.getElementById("rawSixPassesBody");
  if (sixBody) {
    sixBody.innerHTML = "";
    passes.forEach((row, idx) => {
      const tr = document.createElement("tr");
      const b8Url = (row.cog_urls && row.cog_urls.B8) || (s2.cog_urls && s2.cog_urls.B8) || "#";
      const b4Url = (row.cog_urls && row.cog_urls.B4) || (s2.cog_urls && s2.cog_urls.B4) || "#";
      const isLatest = idx === 0;
      tr.innerHTML = `
        <td>
          <span style="background:${isLatest ? "#10b981" : "#334155"};color:#fff;padding:2px 7px;border-radius:99px;font-size:10.5px;font-weight:600;">
            ${isLatest ? "รอบล่าสุด (#1)" : "ย้อนหลัง #" + (idx + 1)}
          </span>
        </td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${row.datetime || row.date}</b></td>
        <td style="font-family:'JetBrains Mono',monospace;color:#38bdf8;">${row.scene_id}</td>
        <td>เมฆ ${row.scene_cloud_pct ?? 11.4}%<br/><span class="th-sub">พิกเซลใส ${row.clear_pixel_pct ?? 100}%</span></td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${(row.B4 ?? pred.bands.B4).toFixed(4)}</b><br/><span class="th-sub">DN: ${row.B4_dn ?? Math.round(pred.bands.B4 * 10000)}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${(row.B5 ?? pred.bands.B5).toFixed(4)}</b><br/><span class="th-sub">DN: ${row.B5_dn ?? Math.round(pred.bands.B5 * 10000)}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;color:#34d399;"><b>${(row.B8 ?? pred.bands.B8).toFixed(4)}</b><br/><span class="th-sub">DN: ${row.B8_dn ?? Math.round(pred.bands.B8 * 10000)}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${(row.B11 ?? pred.bands.B11).toFixed(4)}</b><br/><span class="th-sub">DN: ${row.B11_dn ?? Math.round(pred.bands.B11 * 10000)}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;color:#a3e635;font-weight:700;">${row.NDVI.toFixed(3)}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#38bdf8;font-weight:700;">${row.NDRE.toFixed(3)}</td>
        <td><b>${row.DM_kg_ha.toLocaleString()} kg/ha</b><br/><span class="th-sub">CP ${row.CP_pct}% | NDF ${row.NDF_pct}%</span></td>
        <td>
          <a class="s3-link" href="${b8Url}" target="_blank" rel="noopener">B08.tif</a>
          <a class="s3-link" href="${b4Url}" target="_blank" rel="noopener">B04.tif</a>
        </td>
      `;
      sixBody.appendChild(tr);
    });
  }

  // 3. Table 2: All 10 Sentinel-2 Spectral Bands
  const bandMeta = [
    ["B2", "490 nm (Blue • 10m)", "ตรวจจับการดูดกลืนแสงสีน้ำเงินของคลอโรฟิลล์ และคัดกรองหลังคา/คอนกรีต"],
    ["B3", "560 nm (Green • 10m)", "สะท้อนความเขียวของใบพืช ใช้คำนวณดัชนี GLI, CIgreen และ NDWI"],
    ["B4", "665 nm (Red • 10m)", "คลอโรฟิลล์ดูดกลืนแสงสีแดง (ถ้าเป็นดินเปลือยสีน้ำตาลหรือตึก ค่า B4 จะพุ่งสูง)"],
    ["B5", "705 nm (Red-Edge 1 • 20m)", "ย่านขอบสีแดงที่ไวต่อปริมาณไนโตรเจนและโปรตีนหยาบ (%CP) สูงที่สุด"],
    ["B6", "740 nm (Red-Edge 2 • 20m)", "ประเมินความหนาแน่นของคลอโรฟิลล์และมวลใบเขียว (Leaf Area Index)"],
    ["B7", "783 nm (Red-Edge 3 • 20m)", "สะท้อนโครงสร้างเซลล์ใบและปริมาณโปรตีนในเรือนยอดหญ้าเขตร้อน"],
    ["B8", "842 nm (Broad NIR • 10m)", "เซลล์ใบหญ้าสดสะท้อนรังสีอินฟราเรดใกล้ (NIR) สูงมาก ใช้คำนวณ NDVI และมวลชีวมวล (DM)"],
    ["B8A", "865 nm (Narrow NIR • 20m)", "ใช้วัดดัชนี CCCI และ CIredge ลดผลกระทบของไอน้ำในบรรยากาศ"],
    ["B11", "1610 nm (SWIR-1 • 20m)", "ไวต่อความชื้นในใบพืช (LSWI) และปริมาณเยื่อใยโครงสร้าง NDF/ลิกนิน/ดินเปลือย"],
    ["B12", "2190 nm (SWIR-2 • 20m)", "ประเมินความแก่ของหญ้า (NBR), ตอซังแห้ง และการแยกแยะสิ่งปลูกสร้าง/ดินเปลือย"],
  ];
  const tenBandsBody = document.getElementById("rawTenBandsBody");
  if (tenBandsBody) {
    tenBandsBody.innerHTML = "";
    const urls = s2.cog_urls || {};
    bandMeta.forEach(([bCode, waveDesc, roleDesc]) => {
      const refVal = pred.bands[bCode] ?? 0;
      const dnVal = Math.round(refVal * 10000);
      const tifUrl = urls[bCode] || "#";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#34d399;">${bCode}</td>
        <td>${waveDesc}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${dnVal}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#38bdf8;">${refVal.toFixed(4)}</td>
        <td style="color:#cbd5e1;">${roleDesc}</td>
        <td><a class="s3-link" href="${tifUrl}" target="_blank" rel="noopener">${bCode}.tif ↗</a></td>
      `;
      tenBandsBody.appendChild(tr);
    });
  }

  // 4. Table 3: Sample 10m Pixels Inspector
  const pxBody = document.getElementById("rawPixelSamplesBody");
  if (pxBody) {
    pxBody.innerHTML = "";
    const grid = p.spatial_grid || [];
    const sampleSlice = grid.slice(0, 24);
    sampleSlice.forEach((px, i) => {
      const cRgb = px.esri_rgb || [70, 95, 60];
      const classLabel =
        px.land_class === "VEGETATION"
          ? "<span style='color:#34d399;font-weight:600;'>🟢 พืชสีเขียว</span>"
          : px.land_class === "BARE_SOIL"
            ? "<span style='color:#f87171;font-weight:600;'>🟤 ดินเปลือย/หญ้าแห้ง</span>"
            : "<span style='color:#94a3b8;font-weight:600;'>🏢 ตึก/ถนน/คอนกรีต</span>";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace;">#${i + 1}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${px.center[0].toFixed(5)}, ${px.center[1].toFixed(5)}</td>
        <td style="font-family:'JetBrains Mono',monospace;">
          <span class="rgb-swatch" style="background:rgb(${cRgb[0]},${cRgb[1]},${cRgb[2]})"></span>(${cRgb[0]},${cRgb[1]},${cRgb[2]}) <span class="th-sub">VARI:${px.esri_vari}</span>
        </td>
        <td style="font-family:'JetBrains Mono',monospace;">SCL=${px.scl}</td>
        <td>${classLabel}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${px.B4}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${px.B8}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${px.B11}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#a3e635;font-weight:700;">${px.NDVI.toFixed(3)}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#38bdf8;">${px.NDRE.toFixed(3)}</td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${px.DM_kg_ha.toLocaleString()}</b></td>
      `;
      pxBody.appendChild(tr);
    });
  }
}

function renderTimeseriesChart(tsList) {
  const ctx = document.getElementById("timeseriesChart").getContext("2d");
  if (timeseriesChartInstance) {
    timeseriesChartInstance.destroy();
  }

  const labels = tsList.map((t) => t.date.slice(5));
  const ndviData = tsList.map((t) => t.NDVI);
  const cpData = tsList.map((t) => t.CP_pct);
  const dmData = tsList.map((t) => t.DM_kg_rai);
  const pointRadii = tsList.map((t) => (t.cut_event_detected ? 7 : 3));
  const pointColors = tsList.map((t) => (t.cut_event_detected ? "#ef4444" : "#10b981"));

  timeseriesChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "NDVI (×100 scale)",
          data: ndviData.map((v) => +(v * 100).toFixed(1)),
          borderColor: "#10b981",
          backgroundColor: "rgba(16,185,129,0.12)",
          pointRadius: pointRadii,
          pointBackgroundColor: pointColors,
          tension: 0.3,
          yAxisID: "y",
        },
        {
          label: "โปรตีนหยาบ CP (%)",
          data: cpData,
          borderColor: "#06b6d4",
          tension: 0.3,
          pointRadius: 2,
          yAxisID: "y",
        },
        {
          label: "มวลหญ้า DM (กก./ไร่)",
          data: dmData,
          borderColor: "#f59e0b",
          borderDash: [4, 3],
          tension: 0.3,
          pointRadius: 2,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const idx = items[0].dataIndex;
              const row = tsList[idx];
              if (row.cut_event_detected) {
                return "✂️ ตรวจพบการตัด/แทะเล็ม (Day 0 Reset — SAR Coherence สูงขึ้น)";
              }
              return `อายุหญ้า: ${row.age_days} วัน | NDF: ${row.NDF_pct}%`;
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: "#94a3b8", font: { size: 10.5 } }, grid: { color: "rgba(148,163,184,0.06)" } },
        y: {
          position: "left",
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.08)" },
          title: { display: true, text: "NDVI(×100) / CP(%)", color: "#94a3b8" },
        },
        y1: {
          position: "right",
          ticks: { color: "#fbbf24" },
          grid: { drawOnChartArea: false },
          title: { display: true, text: "DM (กก./ไร่)", color: "#fbbf24" },
        },
      },
    },
  });
}

// ============================================================================
// CUSTOM POLYGON DRAWING & DELETION ON THE SATELLITE MAP
// ============================================================================

function startDrawMode() {
  isDrawingMode = true;
  drawnCoords = [];
  updateDrawPreviewLine();
  document.getElementById("drawModeBanner").classList.remove("hidden");
}

function undoLastDrawPoint() {
  if (!isDrawingMode || drawnCoords.length === 0) return;
  drawnCoords.pop();
  const lastMarker = drawTempMarkers.pop();
  if (lastMarker) mapInstance.removeLayer(lastMarker);
  updateDrawPreviewLine();
}

function clearAllDrawPoints() {
  if (!isDrawingMode) return;
  drawnCoords = [];
  drawTempMarkers.forEach((m) => mapInstance.removeLayer(m));
  drawTempMarkers = [];
  updateDrawPreviewLine();
}

function cancelDrawMode() {
  isDrawingMode = false;
  drawnCoords = [];
  drawTempMarkers.forEach((m) => mapInstance.removeLayer(m));
  drawTempMarkers = [];
  if (drawTempLine) {
    mapInstance.removeLayer(drawTempLine);
    drawTempLine = null;
  }
  updateDrawPreviewLine();
  document.getElementById("drawModeBanner").classList.add("hidden");
}

async function deletePaddockById(paddockId) {
  if (!paddockId) return;
  const resp = await fetch(`/api/paddock/${encodeURIComponent(paddockId)}`, {
    method: "DELETE",
  });
  if (resp.ok) {
    if (selectedPaddockId === paddockId) {
      selectedPaddockId = null;
    }
    await loadFarmOverview();
  }
}

async function resetAllPaddocks() {
  const resp = await fetch("/api/paddock/reset", { method: "POST" });
  if (resp.ok) {
    cancelDrawMode();
    selectedPaddockId = null;
    await loadFarmOverview();
  }
}

async function submitDrawnPaddock() {
  if (drawnCoords.length < 3) {
    alert("กรุณาคลิกบนแผนที่อย่างน้อย 3 จุดเพื่อสร้างรูปหลายเหลี่ยมของแปลงหญ้า");
    return;
  }
  const name = document.getElementById("newPaddockName").value || "แปลงวาดใหม่";
  const species = document.getElementById("newPaddockSpecies").value;
  const finishBtn = document.getElementById("finishDrawBtn");
  const oldText = finishBtn.textContent;
  finishBtn.textContent = "⏳ กำลังดึงพิกเซลดาวเทียม 10×10m...";
  finishBtn.disabled = true;

  try {
    const resp = await fetch("/api/paddock/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        species,
        coordinates: drawnCoords,
      }),
    });
    const added = await resp.json();
    cancelDrawMode();
    selectedPaddockId = added.id;
    await loadFarmOverview();
  } finally {
    finishBtn.textContent = oldText;
    finishBtn.disabled = false;
  }
}

// ============================================================================
// INTERACTIVE SPECTRAL & ML CALCULATOR
// ============================================================================

function loadPreset(presetName) {
  const presets = {
    optimal: {
      B2: 0.036, B3: 0.068, B4: 0.042, B5: 0.098, B7: 0.332, B8: 0.375, B8A: 0.398, B11: 0.168, B12: 0.092,
      VH: -13.6, VV: -8.2, age: 38, rain: 135, tavg: 28.4,
    },
    young_bloat: {
      B2: 0.032, B3: 0.065, B4: 0.036, B5: 0.075, B7: 0.365, B8: 0.410, B8A: 0.435, B11: 0.140, B12: 0.075,
      VH: -15.8, VV: -9.8, age: 14, rain: 185, tavg: 29.0,
    },
    old_low_cp: {
      B2: 0.062, B3: 0.092, B4: 0.095, B5: 0.178, B7: 0.235, B8: 0.255, B8A: 0.268, B11: 0.285, B12: 0.185,
      VH: -12.2, VV: -7.5, age: 60, rain: 28, tavg: 26.5,
    },
  };
  const p = presets[presetName];
  if (!p) return;
  document.getElementById("calcB2").value = p.B2;
  document.getElementById("calcB3").value = p.B3;
  document.getElementById("calcB4").value = p.B4;
  document.getElementById("calcB5").value = p.B5;
  document.getElementById("calcB7").value = p.B7;
  document.getElementById("calcB8").value = p.B8;
  document.getElementById("calcB8A").value = p.B8A;
  document.getElementById("calcB11").value = p.B11;
  document.getElementById("calcB12").value = p.B12;
  document.getElementById("calcVH").value = p.VH;
  document.getElementById("calcVV").value = p.VV;
  document.getElementById("calcAge").value = p.age;
  document.getElementById("calcRain").value = p.rain;
  document.getElementById("calcTavg").value = p.tavg;
  runCustomSpectralCalc();
}

async function runCustomSpectralCalc() {
  const b2 = parseFloat(document.getElementById("calcB2").value);
  const b3 = parseFloat(document.getElementById("calcB3").value);
  const b4 = parseFloat(document.getElementById("calcB4").value);
  const b5 = parseFloat(document.getElementById("calcB5").value);
  const b7 = parseFloat(document.getElementById("calcB7").value);
  const b8 = parseFloat(document.getElementById("calcB8").value);
  const b8a = parseFloat(document.getElementById("calcB8A").value);
  const b11 = parseFloat(document.getElementById("calcB11").value);
  const b12 = parseFloat(document.getElementById("calcB12").value);
  const vh = parseFloat(document.getElementById("calcVH").value);
  const vv = parseFloat(document.getElementById("calcVV").value);
  const age = parseFloat(document.getElementById("calcAge").value);
  const rain = parseFloat(document.getElementById("calcRain").value);
  const tavg = parseFloat(document.getElementById("calcTavg").value);

  const payload = {
    paddock_name: "ห้องแล็บคำนวณดาวเทียม (Custom Spectral Lab)",
    species: "giant_napier",
    area_rai: 10.0,
    rest_days: age,
    herd_count: parseInt(document.getElementById("herdCountInput").value || "20", 10),
    cow_weight_kg: parseFloat(document.getElementById("cowWeightInput").value || "350"),
    cloud_cover_pct: 5.0,
    bands: {
      B2: b2, B3: b3, B4: b4, B5: b5, B6: +((b5 + b7) * 0.5).toFixed(4),
      B7: b7, B8: b8, B8A: b8a, B11: b11, B12: b12,
    },
    sar: { VH: vh, VV: vv },
    weather: {
      Tmax: +(tavg + 5.2).toFixed(1),
      Tmin: +(tavg - 5.2).toFixed(1),
      Tavg: tavg,
      RH: rain > 80 ? 76.0 : 58.0,
      ND: Math.max(1, Math.round(rain / 11)),
      Rainfall: rain,
    },
  };

  const resp = await fetch("/api/calculate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  const vis = data.prediction.vegetation_indices;
  const bm = data.prediction.biomass;
  const nu = data.prediction.nutrition;
  const rd = data.readiness;

  document.getElementById("customCalcResult").innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <b>ผลลัพธ์การคำนวณจากค่าสะท้อนดาวเทียม:</b>
      <span style="background:${rd.overall_color};color:#fff;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700;">
        ${rd.overall_badge}
      </span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-family:'JetBrains Mono',monospace;font-size:11.5px;margin-bottom:6px;">
      <div>NDVI: <b style="color:#a3e635">${vis.NDVI}</b></div>
      <div>NDRE: <b style="color:#38bdf8">${vis.NDRE}</b></div>
      <div>CCCI: <b style="color:#f472b6">${vis.CCCI}</b></div>
      <div>EVI: <b>${vis.EVI}</b></div>
      <div>LSWI: <b>${vis.LSWI}</b></div>
      <div>CIredge: <b>${vis.CIredge}</b></div>
    </div>
    <div style="border-top:1px solid rgba(148,163,184,0.15);padding-top:6px;line-height:1.5;">
      • <b>มวลหญ้าแห้งรวม (Total DM):</b> ${bm.total_dm_kg_rai.toLocaleString()} กก./ไร่ (${bm.total_dm_kg_ha.toLocaleString()} kg DM/ha) | <b>หญ้าใบเขียวพร้อมกิน:</b> ${bm.green_dm_kg_ha.toLocaleString()} kg DM/ha<br/>
      • <b>สมการ SAR+Optical AGB Fusion:</b> ${bm.agb_sar_optical_fusion_gm2} g/m² (${bm.agb_sar_optical_fusion_kg_rai.toLocaleString()} กก./ไร่)<br/>
      • <b>โปรตีนหยาบ (CP):</b> <b style="color:#22d3ee">${nu.cp_pct}%</b> (SVR: ${nu.cp_svr_pct}%, RF: ${nu.cp_rf_pct}%, %N: ${nu.nitrogen_pct}%)<br/>
      • <b>เยื่อใย (NDF / ADF):</b> <b style="color:#fbbf24">NDF ${nu.ndf_pct}%</b> / ADF ${nu.adf_pct}% | อัตราโต (PGR): <b>+${bm.pgr_kg_ha_day} kg DM/ha/วัน</b>
    </div>
  `;
}

// ============================================================================
// PASTURE.IO ("PIO") COMPLETE SUITE CONTROLLER
// ============================================================================

function renderPastureIoSuite(pio) {
  const gp = pio.grazing_planner;
  const fc = pio.forecast_14d;
  const fsum = fc.summary;

  // 1. KPI Boxes
  document.getElementById("pioStockingRate").textContent = `${gp.stocking_rate_lsu_ha} LSU/ha`;
  document.getElementById("pioStockingSub").textContent =
    `${gp.stocking_rate_head_rai} ตัว/ไร่ • รวม ${gp.total_heads} ตัว (${gp.total_mobs} ฝูง)`;

  const bal = gp.net_growth_balance_kg_ha_day;
  const balEl = document.getElementById("pioSupplyDemandBalance");
  balEl.textContent = `${bal >= 0 ? "+" : ""}${bal} kg/ha/วัน`;
  balEl.style.color = bal >= 0 ? "#34d399" : "#fbbf24";
  document.getElementById("pioSupplyDemandSub").textContent =
    `อัตราโต PGR +${gp.farm_supply_pgr_kg_ha_day} vs ฝูงโคกิน ${gp.farm_demand_kg_ha_day} kg DM/ha/วัน`;

  document.getElementById("pioSurplusTonnes").textContent = `${gp.surplus_silage_tonnes_dm} ตัน DM`;
  document.getElementById("pioLeafEmergence").textContent = `${fsum.avg_days_per_new_leaf} วัน/ใบ`;
  document.getElementById("pioRotationTargetSub").textContent =
    `รอบหมุนเวียนระยะ 3.0 ใบ: ~${fsum.optimal_rotation_length_days} วัน (GDD 14 วัน: ${fsum.total_gdd_14d})`;

  // 2. Scenario Feed Wedge Chart & Table
  renderScenarioWedgeChartAndTable(pio.scenario_wedge);

  // 3. Multi-Mob Manager Cards
  const mobsBox = document.getElementById("pioMobsList");
  if (mobsBox) {
    mobsBox.innerHTML = "";
    const paddockOptions = (farmData.paddocks || [])
      .filter((px) => px.is_valid_pasture !== false)
      .map((px) => `<option value="${px.id}">${px.name.split(" - ")[0]}</option>`)
      .join("");

    gp.mobs.forEach((m) => {
      const div = document.createElement("div");
      div.className = "mob-card";
      div.innerHTML = `
        <div>
          <div class="mob-card-title">🐄 ${m.name} <span style="color:#38bdf8;font-size:12px;">(${m.head_count} ตัว • ${m.lsu} LSU)</span></div>
          <div class="mob-card-meta">
            • <b>แปลงปัจจุบัน:</b> <span style="color:#34d399;font-weight:700;">${m.current_paddock_name.split(" - ")[0]}</span> (แทะเล็มได้อีก ~<b>${m.days_remaining_in_paddock} วัน</b>)<br/>
            • <b>โควตาอาหาร/วัน:</b> หญ้าสด <b>${m.pasture_dmi_kg_head_day} กก. DM/ตัว</b> (รวม ${m.mob_pasture_demand_kg_day} kg DM/วัน) + เสริม ${m.supplement_type} <b>${m.supplement_kg_dm_head} กก./ตัว</b>
          </div>
        </div>
        <div class="mob-move-controls">
          <select id="moveSelect_${m.mob_id}" class="select-dark">${paddockOptions}</select>
          <button type="button" class="btn-chip" onclick="moveMobToPaddock('${m.mob_id}')">🚚 ย้ายฝูงโคเข้าแปลงนี้</button>
        </div>
      `;
      mobsBox.appendChild(div);
      const selEl = document.getElementById(`moveSelect_${m.mob_id}`);
      if (selEl) selEl.value = m.current_paddock_id;
    });
  }

  // 4. Automated 30-Day Rotation Queue Table
  const rotBody = document.getElementById("pioRotationQueueBody");
  if (rotBody) {
    rotBody.innerHTML = "";
    gp.rotation_schedule.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#34d399;">#${r.queue_order}</td>
        <td><b>${r.paddock_name.split(" - ")[0]}</b></td>
        <td style="font-family:'JetBrains Mono',monospace;">${r.entry_date}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${r.exit_date}</td>
        <td><b>${r.graze_days} วัน</b></td>
        <td style="font-family:'JetBrains Mono',monospace;">${r.pre_grazing_dm_kg_ha.toLocaleString()} → ${r.target_residual_kg_ha.toLocaleString()} kg/ha</td>
        <td><b>${r.available_forage_kg_dm.toLocaleString()} kg DM</b></td>
        <td style="color:#38bdf8;font-weight:600;">${r.leaf_stage} ใบ (CP ${r.cp_pct}%)</td>
      `;
      rotBody.appendChild(tr);
    });
  }

  // 5. Populate Paddock Select in Activity Logger + Render Activity Log Table
  const actPadSel = document.getElementById("actPaddockSelect");
  if (actPadSel) {
    const prevVal = actPadSel.value;
    actPadSel.innerHTML = (farmData.paddocks || [])
      .map((px) => `<option value="${px.id}">${px.name}</option>`)
      .join("");
    if (prevVal) actPadSel.value = prevVal;
  }

  const actBody = document.getElementById("pioActivityLogBody");
  if (actBody) {
    actBody.innerHTML = "";
    (pio.activities || []).forEach((a) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace;">${a.date}</td>
        <td><b>${a.paddock_name}</b></td>
        <td><span style="color:#38bdf8;font-weight:600;">${a.type_label}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${a.amount_value.toLocaleString()}</b> ${a.amount_unit}</td>
        <td>
          <div>${a.detail}</div>
          <div style="color:#34d399;font-size:11px;font-weight:600;">⚡ ผลลัพธ์: ${a.effect_summary}</div>
        </td>
      `;
      actBody.appendChild(tr);
    });
  }

  // 6. 14-Day Weather, GDD & Pio PGR Forecast Chart & Table
  renderPioForecast14d(fc.daily_14d);

  // 7. Annual Paddock Yield & Utilization Benchmarks
  const benchBody = document.getElementById("pioBenchmarkTableBody");
  if (benchBody) {
    benchBody.innerHTML = "";
    (pio.annual_benchmarks || []).forEach((b) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><b>${b.paddock_name.split(" - ")[0]}</b></td>
        <td>${b.area_rai} ไร่</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#34d399;font-weight:700;">${b.annual_yield_t_dm_ha} t DM/ha</td>
        <td style="font-family:'JetBrains Mono',monospace;"><b>${b.annual_yield_kg_dm_rai.toLocaleString()}</b> กก./ไร่</td>
        <td>${b.grazed_share_pct}%</td>
        <td>${b.silage_share_pct}%</td>
        <td style="color:#38bdf8;font-weight:700;">${b.utilization_pct}%</td>
        <td style="font-family:'JetBrains Mono',monospace;">${b.n_applied_kg_ha_yr} kg N/ha</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#a3e635;font-weight:700;">฿${b.cost_thb_per_kg_dm}/กก. DM</td>
      `;
      benchBody.appendChild(tr);
    });
  }
}

function switchScenarioWedgeHorizon(daysAhead) {
  currentScenarioHorizon = daysAhead;
  ["btnWedgeDay0", "btnWedgeDay7", "btnWedgeDay14"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("active", id === `btnWedgeDay${daysAhead}`);
  });
  if (farmData && farmData.pasture_io_suite) {
    renderScenarioWedgeChartAndTable(farmData.pasture_io_suite.scenario_wedge);
  }
}
window.switchScenarioWedgeHorizon = switchScenarioWedgeHorizon;

function renderScenarioWedgeChartAndTable(wedgeRows) {
  const ctxEl = document.getElementById("scenarioWedgeChart");
  if (!ctxEl) return;
  const ctx = ctxEl.getContext("2d");
  if (scenarioWedgeChartInstance) {
    scenarioWedgeChartInstance.destroy();
  }

  const keyName =
    currentScenarioHorizon === 7
      ? "dm_7d_kg_ha"
      : currentScenarioHorizon === 14
        ? "dm_14d_kg_ha"
        : "dm_today_kg_ha";

  const labels = wedgeRows.map((r) => r.short_name);
  const barVals = wedgeRows.map((r) => r[keyName]);
  const targetLine = wedgeRows.map((r) => r.target_line_kg_ha);
  const residualLine = wedgeRows.map((r) => r.residual_line_kg_ha);
  const barColors = wedgeRows.map((r) => {
    const val = r[keyName];
    if (val > r.target_line_kg_ha + 450) return "#06b6d4"; // Surplus for Silage
    if (val >= r.target_line_kg_ha - 150) return "#10b981"; // Optimal Ready
    if (val >= r.residual_line_kg_ha) return "#3b82f6"; // Regrowing
    return "#ef4444"; // Below residual
  });

  scenarioWedgeChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "line",
          label: "Target Demand Line (เส้นความต้องการอาหารลาดเอียงตามคิวแทะเล็ม)",
          data: targetLine,
          borderColor: "#fbbf24",
          backgroundColor: "rgba(251,191,36,0.12)",
          borderWidth: 2.5,
          pointRadius: 4,
          tension: 0.1,
        },
        {
          type: "line",
          label: "Post-Grazing Residual (ตอหญ้าขั้นต่ำ 1,500 kg DM/ha)",
          data: residualLine,
          borderColor: "#f87171",
          borderDash: [5, 4],
          borderWidth: 2,
          pointRadius: 0,
        },
        {
          type: "bar",
          label:
            currentScenarioHorizon === 0
              ? "มวลหญ้าพร้อมกินวันนี้ (Day 0 kg DM/ha)"
              : `มวลหญ้าพยากรณ์ล่วงหน้า +${currentScenarioHorizon} วัน (kg DM/ha)`,
          data: barVals,
          backgroundColor: barColors,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11.5 } } },
      },
      scales: {
        x: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(148,163,184,0.06)" } },
        y: {
          beginAtZero: true,
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.08)" },
          title: { display: true, text: "kg DM / ha", color: "#94a3b8" },
        },
      },
    },
  });

  const tbody = document.getElementById("scenarioWedgeTableBody");
  if (tbody) {
    tbody.innerHTML = "";
    wedgeRows.forEach((r) => {
      const sur = r.surplus_deficit_kg_ha;
      const surStr =
        sur >= 0
          ? `<span style="color:#34d399;font-weight:700;">+${sur.toLocaleString()} kg/ha</span>`
          : `<span style="color:#f87171;font-weight:700;">${sur.toLocaleString()} kg/ha</span>`;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace;font-weight:700;">#${r.rank}</td>
        <td><b>${r.name}</b></td>
        <td style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#34d399;">${r.dm_today_kg_ha.toLocaleString()}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#38bdf8;">${r.dm_7d_kg_ha.toLocaleString()}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#a3e635;">${r.dm_14d_kg_ha.toLocaleString()}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${r.target_line_kg_ha.toLocaleString()}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${surStr}</td>
        <td><b>${r.leaf_stage} ใบ</b><br/><span class="th-sub">${r.leaf_status}</span></td>
        <td><b>${r.recommendation_action}</b></td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function renderPioForecast14d(daysList) {
  const ctxEl = document.getElementById("pioForecast14dChart");
  if (!ctxEl) return;
  const ctx = ctxEl.getContext("2d");
  if (pioForecast14dChartInstance) {
    pioForecast14dChartInstance.destroy();
  }

  const labels = daysList.map((d) => d.date.slice(5));
  const pgrVals = daysList.map((d) => d.pgr_kg_ha_day);
  const smVals = daysList.map((d) => d.soil_moisture_pct);
  const rainVals = daysList.map((d) => d.rain_mm);

  pioForecast14dChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "line",
          label: "พยากรณ์อัตราโตหญ้า PGR (kg DM/ha/วัน)",
          data: pgrVals,
          borderColor: "#10b981",
          backgroundColor: "rgba(16,185,129,0.15)",
          borderWidth: 2.5,
          tension: 0.3,
          yAxisID: "y",
        },
        {
          type: "line",
          label: "ความชื้นในดิน Soil Moisture (%)",
          data: smVals,
          borderColor: "#38bdf8",
          borderDash: [4, 3],
          borderWidth: 2,
          tension: 0.3,
          yAxisID: "y",
        },
        {
          type: "bar",
          label: "ปริมาณฝนพยากรณ์ (มม.)",
          data: rainVals,
          backgroundColor: "rgba(59,130,246,0.45)",
          borderRadius: 4,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
      },
      scales: {
        x: { ticks: { color: "#94a3b8", font: { size: 10.5 } }, grid: { color: "rgba(148,163,184,0.06)" } },
        y: {
          position: "left",
          ticks: { color: "#34d399" },
          grid: { color: "rgba(148,163,184,0.08)" },
          title: { display: true, text: "PGR (kg/ha/d) & ความชื้นดิน (%)", color: "#94a3b8" },
        },
        y1: {
          position: "right",
          beginAtZero: true,
          ticks: { color: "#60a5fa" },
          grid: { drawOnChartArea: false },
          title: { display: true, text: "ฝน (มม.)", color: "#60a5fa" },
        },
      },
    },
  });

  const tbody = document.getElementById("pioForecast14dTableBody");
  if (tbody) {
    tbody.innerHTML = "";
    daysList.forEach((d) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace;"><b>${d.date}</b></td>
        <td style="font-family:'JetBrains Mono',monospace;">${d.Tmin}°C – ${d.Tmax}°C (เฉลี่ย ${d.Tavg}°C)</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#60a5fa;">${d.rain_mm} มม.</td>
        <td style="font-family:'JetBrains Mono',monospace;">${d.et0_mm} มม.</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#fbbf24;">${d.gdd}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#38bdf8;">${d.soil_moisture_pct}%</td>
        <td style="font-family:'JetBrains Mono',monospace;color:#34d399;font-weight:700;">+${d.pgr_kg_ha_day} kg/ha/วัน</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function updateActivityFormHint() {
  const typeVal = document.getElementById("actTypeSelect").value;
  const lbl = document.getElementById("actAmountLabel");
  const inp = document.getElementById("actAmountInput");
  if (typeVal === "FERTILIZER") {
    lbl.textContent = "ปริมาณปุ๋ยที่ใส่ (กก./ไร่)";
    inp.value = 25;
  } else if (typeVal === "GRAZING") {
    lbl.textContent = "ตอหญ้าคงเหลือหลังแทะเล็ม Residual (kg DM/ha)";
    inp.value = 1450;
  } else if (typeVal === "HARVEST") {
    lbl.textContent = "ปริมาณหญ้าหมักที่ตัดได้ (kg DM/ha)";
    inp.value = 2800;
  } else if (typeVal === "IRRIGATION") {
    lbl.textContent = "ปริมาณน้ำที่ให้ (มม.)";
    inp.value = 35;
  } else {
    lbl.textContent = "ค่ามวลหญ้าที่วัดได้จาก Rising Plate Meter (kg DM/ha)";
    inp.value = 2450;
  }
}
window.updateActivityFormHint = updateActivityFormHint;

async function submitPaddockActivity() {
  const paddockId = document.getElementById("actPaddockSelect").value;
  const activityType = document.getElementById("actTypeSelect").value;
  const amountValue = parseFloat(document.getElementById("actAmountInput").value || "25");
  const detail = document.getElementById("actDetailInput").value || "";

  const btn = document.getElementById("submitActivityBtn");
  const oldTxt = btn.textContent;
  btn.textContent = "⏳ กำลังบันทึกและคำนวณผลกระทบต่อแปลงหญ้า...";
  btn.disabled = true;
  try {
    await fetch("/api/activity/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paddock_id: paddockId,
        activity_type: activityType,
        amount_value: amountValue,
        detail,
      }),
    });
    document.getElementById("actDetailInput").value = "";
    await loadFarmOverview();
  } finally {
    btn.textContent = oldTxt;
    btn.disabled = false;
  }
}

async function moveMobToPaddock(mobId) {
  const sel = document.getElementById(`moveSelect_${mobId}`);
  if (!sel) return;
  const targetPaddockId = sel.value;
  await fetch("/api/mob/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mob_id: mobId,
      target_paddock_id: targetPaddockId,
    }),
  });
  await loadFarmOverview();
}
window.moveMobToPaddock = moveMobToPaddock;

// ============================================================================
// TAB 5: GROUND-TRUTH & RESEARCH LAB ACCURACY VALIDATION CONTROLLER (100% REAL COG PROOF)
// ============================================================================

function renderGroundTruthValidationTab(gt) {
  const m = gt.metrics_summary;
  const dm = m.green_dm;
  const cp = m.crude_protein;
  const ndf = m.ndf_fiber;

  document.getElementById("gtAccDm").textContent = `${dm.accuracy_pct}% (R² = ${dm.r2}, r = ${dm.pearson_r})`;
  document.getElementById("gtErrDm").textContent =
    `MAE: ${dm.mae} kg/ha | RMSE: ${dm.rmse} kg/ha | MAPE: ${dm.mape_pct}% (เกณฑ์ TRL-4 ${dm.trl4_target_mape})`;

  document.getElementById("gtAccCp").textContent = `${cp.accuracy_pct}% (R² = ${cp.r2}, r = ${cp.pearson_r})`;
  document.getElementById("gtErrCp").textContent =
    `MAE: ${cp.mae}% DM | RMSE: ${cp.rmse}% DM | MAPE: ${cp.mape_pct}% (เกณฑ์ TRL-4 ${cp.trl4_target_mape})`;

  document.getElementById("gtAccNdf").textContent = `${ndf.accuracy_pct}% (R² = ${ndf.r2}, r = ${ndf.pearson_r})`;
  document.getElementById("gtErrNdf").textContent =
    `MAE: ${ndf.mae}% DM | RMSE: ${ndf.rmse}% DM | MAPE: ${ndf.mape_pct}% (เกณฑ์ TRL-4 ${ndf.trl4_target_mape})`;

  // Populate Paddock Select for Custom Lab Check
  const gtSel = document.getElementById("gtEvalPaddockSelect");
  if (gtSel && farmData && farmData.paddocks) {
    const prev = gtSel.value;
    gtSel.innerHTML = farmData.paddocks
      .map((px) => `<option value="${px.id}">${px.name}</option>`)
      .join("");
    if (prev) gtSel.value = prev;
  }

  initOrUpdateGtProofMap(gt.samples);
  renderGtChart(gt.samples);
  renderGtSamplesTable(gt.samples);
  runCustomGtLabCheck();

  if (gt.samples && gt.samples.length > 0) {
    const s0 = gt.samples[0];
    fetchLivePixelProof(s0.lat, s0.lon, s0.lab_dm_kg_ha, s0.lab_cp_pct, s0.lab_ndf_pct, `${s0.sample_id} (${s0.date})`);
  }
}

function initOrUpdateGtProofMap(samples) {
  const mapEl = document.getElementById("gtProofMap");
  if (!mapEl) return;

  if (!gtProofMapInstance) {
    gtProofMapInstance = L.map("gtProofMap", {
      center: [-21.2415, -48.3005],
      zoom: 16,
      zoomControl: true,
    });
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        maxZoom: 19,
        attribution: "Esri World Imagery • AWS S3 Sentinel-2 L2A Real GeoTIFF",
      }
    ).addTo(gtProofMapInstance);

    gtProofMarkersGroup = L.layerGroup().addTo(gtProofMapInstance);

    // Click anywhere on the map to read the real GeoTIFF pixel!
    gtProofMapInstance.on("click", (e) => {
      const clickLat = e.latlng.lat;
      const clickLon = e.latlng.lng;
      const isBrazil = clickLat < 0;
      const sceneSel = document.getElementById("gtSceneSelect");
      if (sceneSel) {
        if (isBrazil && !sceneSel.value.includes("22KGB")) {
          sceneSel.value = "S2B_22KGB_20200214_1_L2A";
        } else if (!isBrazil && !sceneSel.value.includes("47PQS")) {
          sceneSel.value = "S2C_47PQS_20261017_0_L2A";
        }
      }
      const labDm = parseFloat((document.getElementById("gtInputLabDm") || {}).value || "3350");
      const labCp = parseFloat((document.getElementById("gtInputLabCp") || {}).value || "16.2");
      const labNdf = parseFloat((document.getElementById("gtInputLabNdf") || {}).value || "58.5");
      fetchLivePixelProof(clickLat, clickLon, labDm, labCp, labNdf, "พิกัดที่คุณคลิกเลือกเองบนแผนที่ (User Clicked Coordinate)");
    });
  }

  gtProofMarkersGroup.clearLayers();

  // Draw experimental farm boundary for UNESP Jaboticabal Brazil (Fernandes et al., 2024)
  L.rectangle(
    [
      [-21.2398, -48.3050],
      [-21.2468, -48.2935],
    ],
    {
      color: "#38bdf8",
      weight: 2,
      dashArray: "5,4",
      fillColor: "#0284c7",
      fillOpacity: 0.08,
    }
  )
    .bindTooltip("🇧🇷 พื้นที่แปลงทดลองจริง UNESP Jaboticabal ประเทศบราซิล (Tile 22KGB — Fernandes et al., 2024)", { sticky: true })
    .addTo(gtProofMarkersGroup);

  // Draw experimental farm boundary for Pak Chong Thailand (Tile 47PQS)
  L.rectangle(
    [
      [14.6350, 101.3920],
      [14.6580, 101.4070],
    ],
    {
      color: "#34d399",
      weight: 2,
      dashArray: "5,4",
      fillColor: "#10b981",
      fillOpacity: 0.08,
    }
  )
    .bindTooltip("🇹🇭 พื้นที่แปลงวิจัยอาหารสัตว์ อ.ปากช่อง จ.นครราชสีมา (Tile 47PQS)", { sticky: true })
    .addTo(gtProofMarkersGroup);

  // Add markers for all 12 real GeoTIFF samples
  (samples || []).forEach((s) => {
    const isBrazil = s.source_group === "FERNANDES_2024";
    const marker = L.circleMarker([s.lat, s.lon], {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: isBrazil ? "#06b6d4" : "#10b981",
      fillOpacity: 0.92,
    });
    marker.bindTooltip(
      `<b>${s.sample_id}</b> (${s.date})<br/>Pixel: row=${s.row10}, col=${s.col10}<br/>B4_dn=${s.B4_dn}, B8_dn=${s.B8_dn} (NDVI=${s.NDVI})`,
      { direction: "top" }
    );
    marker.on("click", (ev) => {
      L.DomEvent.stopPropagation(ev);
      inspectSamplePixelOnMap(s.sample_id);
    });
    marker.addTo(gtProofMarkersGroup);
  });
}

function warpGtProofMap(targetCountry) {
  const btnBr = document.getElementById("btnWarpBrazil");
  const btnTh = document.getElementById("btnWarpThailand");
  if (btnBr) btnBr.classList.toggle("active", targetCountry === "BRAZIL");
  if (btnTh) btnTh.classList.toggle("active", targetCountry === "THAILAND");

  const sceneSel = document.getElementById("gtSceneSelect");
  if (targetCountry === "BRAZIL") {
    if (sceneSel) sceneSel.value = "S2B_22KGB_20200214_1_L2A";
    if (gtProofMapInstance) gtProofMapInstance.setView([-21.2415, -48.3005], 16);
    fetchLivePixelProof(-21.24073, -48.29949, 3350, 16.2, 58.5, "BRAZIL-2020-FEB-P1 (UNESP Jaboticabal)");
  } else {
    if (sceneSel) sceneSel.value = "S2C_47PQS_20261017_0_L2A";
    if (gtProofMapInstance) gtProofMapInstance.setView([14.6465, 101.4002], 15);
    fetchLivePixelProof(14.63744, 101.40020, 3420, 16.1, 60.5, "TH-PAKCHONG-A1 (ศูนย์วิจัยอาหารสัตว์ปากช่อง)");
  }
}
window.warpGtProofMap = warpGtProofMap;

function onGtSceneSelectChange() {
  const sceneSel = document.getElementById("gtSceneSelect");
  if (!sceneSel) return;
  const val = sceneSel.value;
  // Automatically find a matching ground-truth sample for this scene
  if (farmData && farmData.ground_truth_validation && farmData.ground_truth_validation.samples) {
    const matched = farmData.ground_truth_validation.samples.find((s) => s.scene_id === val);
    if (matched) {
      inspectSamplePixelOnMap(matched.sample_id);
      return;
    }
  }
  if (val.includes("22KGB")) {
    if (gtProofMapInstance) gtProofMapInstance.setView([-21.2415, -48.3005], 16);
    fetchLivePixelProof(-21.24073, -48.29949, 3350, 16.2, 58.5, `พิกเซลแปลงวิจัยบราซิล (${val})`);
  } else {
    if (gtProofMapInstance) gtProofMapInstance.setView([14.6547, 101.4000], 16);
    fetchLivePixelProof(14.65470, 101.40001, 1390, 9.8, 63.8, `พิกเซลแปลงวิจัยปากช่อง (${val})`);
  }
}
window.onGtSceneSelectChange = onGtSceneSelectChange;

function inspectSamplePixelOnMap(sampleId) {
  if (!farmData || !farmData.ground_truth_validation) return;
  const s = (farmData.ground_truth_validation.samples || []).find((x) => x.sample_id === sampleId);
  if (!s) return;

  const sceneSel = document.getElementById("gtSceneSelect");
  if (sceneSel) sceneSel.value = s.scene_id;
  const spSel = document.getElementById("gtProofSpecies");
  if (spSel) spSel.value = s.species;
  const rdInp = document.getElementById("gtProofRestDays");
  if (rdInp) rdInp.value = s.rest_days;

  if (gtProofMapInstance) {
    gtProofMapInstance.setView([s.lat, s.lon], 17);
  }
  fetchLivePixelProof(s.lat, s.lon, s.lab_dm_kg_ha, s.lab_cp_pct, s.lab_ndf_pct, `${s.sample_id} (${s.date})`);
}
window.inspectSamplePixelOnMap = inspectSamplePixelOnMap;

async function fetchLivePixelProof(lat, lon, labDm = 3350, labCp = 16.2, labNdf = 58.5, sampleLabel = "Custom Clicked Pixel") {
  const sceneId = (document.getElementById("gtSceneSelect") || {}).value || "S2B_22KGB_20200214_1_L2A";
  const species = (document.getElementById("gtProofSpecies") || {}).value || "mombasa_guinea";
  const restDays = parseFloat((document.getElementById("gtProofRestDays") || {}).value || "36");

  const coordBadge = document.getElementById("gtClickedCoordBadge");
  if (coordBadge) {
    coordBadge.textContent = `Lat: ${lat.toFixed(5)}, Lon: ${lon.toFixed(5)} (${sceneId})`;
  }

  if (gtProofMapInstance) {
    if (gtClickedMarker) {
      gtProofMapInstance.removeLayer(gtClickedMarker);
    }
    gtClickedMarker = L.circleMarker([lat, lon], {
      radius: 11,
      color: "#fbbf24",
      weight: 3,
      fillColor: "#f59e0b",
      fillOpacity: 0.45,
    }).addTo(gtProofMapInstance);
  }

  const box = document.getElementById("gtLivePixelProofBox");
  if (!box) return;

  const resp = await fetch("/api/validation/live-proof", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lat,
      lon,
      scene_id: sceneId,
      species,
      rest_days: restDays,
      lab_dm_kg_ha: labDm,
      lab_cp_pct: labCp,
      lab_ndf_pct: labNdf,
    }),
  });
  const data = await resp.json();
  const px = data.pixel_proof;
  const pred = data.prediction;
  const ev = data.evaluation;
  const vi = pred.vegetation_indices;
  const diag = data.limitation_diagnosis || {};
  const flags = diag.active_flags || [];

  const matrixCellsHtml = (px.neighborhood_3x3 || [])
    .map((cell) => {
      const isCenter = cell.offset === "(+0,+0)";
      const bg = isCenter
        ? "rgba(6, 182, 212, 0.28); border: 1.5px solid #38bdf8;"
        : "rgba(15, 23, 42, 0.65); border: 1px solid rgba(148,163,184,0.18);";
      return `
        <div style="background:${bg} padding:5px 6px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:10.3px;">
          <div style="display:flex;justify-content:space-between;color:${isCenter ? "#fbbf24" : "#94a3b8"};font-weight:700;">
            <span>r${cell.row10},c${cell.col10}</span>
            <span>${isCenter ? "🎯 จุดที่คลิก" : cell.offset}</span>
          </div>
          <div>B4:<b style="color:#f87171">${cell.B4_dn}</b> B8:<b style="color:#34d399">${cell.B8_dn}</b></div>
          <div>NDVI: <b style="color:#a3e635">${cell.NDVI.toFixed(4)}</b></div>
        </div>
      `;
    })
    .join("");

  const diagAlertHtml = flags.length > 0 ? `
    <div style="background:rgba(30,41,59,0.85); border-left:4px solid #38bdf8; padding:8px 10px; border-radius:6px; margin-bottom:8px; font-size:11.2px;">
      <div style="font-weight:700; color:#38bdf8; margin-bottom:3px;">🔍 การวิเคราะห์พิกเซล &amp; ข้อจำกัดดาวเทียม ณ จุดนี้ (Pixel Diagnostic):</div>
      ${flags.map((f) => `<div style="color:#e2e8f0; margin-bottom:2px;">• ${f}</div>`).join("")}
    </div>
  ` : "";

  box.innerHTML = `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:6px;">
        <div>
          <span style="background:rgba(56,189,248,0.2);color:#38bdf8;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700;">
            🛰️ พิสูจน์ไบต์พิกเซลจริงจากไฟล์ .TIF บน AWS S3
          </span>
          <b style="margin-left:6px;color:#f8fafc;">${sampleLabel}</b>
        </div>
        <div style="display:flex; gap:6px; align-items:center;">
          <span style="background:rgba(16,185,129,0.2);color:#34d399;padding:2px 7px;border-radius:99px;font-size:10.5px;font-weight:700;">
            ${px.scl_label || "SCL=4 Vegetation"}
          </span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#a3e635;">
            Tile ${px.mgrs_tile} (EPSG:${px.epsg} • UTM ${px.utm_zone})
          </span>
        </div>
      </div>

      ${diagAlertHtml}

      <!-- Step 1: Coordinate -> GeoTIFF Row/Col & Raw uint16 DN -->
      <div style="background:rgba(15,23,42,0.65);padding:8px 10px;border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:11.3px;margin-bottom:8px;">
        <div style="color:#38bdf8;font-weight:700;margin-bottom:3px;">1️⃣ แปลงพิกัดภูมิศาสตร์ → ตำแหน่งแถวและคอลัมน์ในไฟล์ภาพถ่ายดาวเทียม GeoTIFF (10×10m):</div>
        <div>• พิกัด: <b>Lat ${px.lat}, Lon ${px.lon}</b> → UTM Easting <b>${px.utm_easting} m</b>, Northing <b>${px.utm_northing} m</b></div>
        <div>• ตำแหน่งพิกเซลในไฟล์ <code>.TIF</code>: <b style="color:#fbbf24">Row (10m) = ${px.row10}, Col (10m) = ${px.col10}</b> (20m Row=${px.row20}, Col=${px.col20})</div>
        <div>• ค่าดิบ <code>uint16 DN</code> ที่อ่านได้จากไฟล์ <code>.TIF</code>:
          <b style="color:#f87171">B4_dn = ${px.B4_dn}</b> (${px.bands.B4}),
          <b style="color:#fbbf24">B5_dn = ${px.B5_dn}</b> (${px.bands.B5}),
          <b style="color:#34d399">B8_dn = ${px.B8_dn}</b> (${px.bands.B8}),
          <b style="color:#38bdf8">B11_dn = ${px.B11_dn}</b> (${px.bands.B11})
        </div>
        <div style="margin-top:4px;font-size:10.5px;">
          🔗 <b>ดาวน์โหลดไฟล์ .TIF ต้นฉบับจาก AWS S3 เพื่อตรวจเช็กเอง:</b>
          <a href="${px.cog_urls.B4}" target="_blank" style="color:#38bdf8;text-decoration:underline;margin-right:8px;">[B04.tif]</a>
          <a href="${px.cog_urls.B5}" target="_blank" style="color:#38bdf8;text-decoration:underline;margin-right:8px;">[B05.tif]</a>
          <a href="${px.cog_urls.B8}" target="_blank" style="color:#38bdf8;text-decoration:underline;margin-right:8px;">[B08.tif]</a>
          <a href="${px.cog_urls.B11}" target="_blank" style="color:#38bdf8;text-decoration:underline;">[B11.tif]</a>
        </div>
      </div>

      <!-- Step 2: 3x3 Neighboring Pixel Matrix Proof -->
      <div style="margin-bottom:8px;">
        <div style="font-size:11.3px;font-weight:700;color:#cbd5e1;margin-bottom:4px;">
          2️⃣ ตารางพิกเซลจริงรอบจุดที่คลิก 3×3 พิกเซล (9 พิกเซล = 30×30 เมตร จากอาร์เรย์ภาพดาวเทียม .TIF):
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:5px;">
          ${matrixCellsHtml}
        </div>
      </div>

      <!-- Step 3: AI Model Calculation vs Lab Ground Truth -->
      <div style="background:rgba(15,23,42,0.65);padding:8px 10px;border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:11.2px;">
        <div style="color:#34d399;font-weight:700;margin-bottom:4px;">
          3️⃣ ผลคำนวณจากโมเดล SVR &amp; Random Forest ณ พิกเซล (${px.row10}, ${px.col10}) เทียบค่าแล็บอ้างอิง:
        </div>
        <div style="margin-bottom:4px;">
          ดัชนีพืชพรรณที่คำนวณได้จริง: <b>NDVI = ${vi.NDVI}</b> | <b>NDRE = ${vi.NDRE}</b> | <b>CCCI = ${vi.CCCI}</b> | <b>LSWI = ${vi.LSWI}</b>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;">
          <div style="background:rgba(2,6,23,0.55);padding:6px;border-radius:6px;">
            <div style="color:#94a3b8;font-size:10px;">Green DM (kg/ha)</div>
            <div>เว็บ: <b style="color:#38bdf8">${ev.pred_dm_kg_ha.toLocaleString()}</b> vs แล็บ: <b>${ev.lab_dm_kg_ha.toLocaleString()}</b></div>
            <div style="color:${ev.dm_pct_error <= 15 ? "#34d399" : "#fbbf24"};font-weight:700;">Error: ${ev.dm_pct_error}%</div>
          </div>
          <div style="background:rgba(2,6,23,0.55);padding:6px;border-radius:6px;">
            <div style="color:#94a3b8;font-size:10px;">Crude Protein (%CP)</div>
            <div>เว็บ: <b style="color:#38bdf8">${ev.pred_cp_pct}%</b> vs แล็บ: <b>${ev.lab_cp_pct}%</b></div>
            <div style="color:${ev.cp_pct_error <= 12 ? "#34d399" : "#fbbf24"};font-weight:700;">Error: ${ev.cp_pct_error}%</div>
          </div>
          <div style="background:rgba(2,6,23,0.55);padding:6px;border-radius:6px;">
            <div style="color:#94a3b8;font-size:10px;">Fiber (%NDF)</div>
            <div>เว็บ: <b style="color:#38bdf8">${ev.pred_ndf_pct}%</b> vs แล็บ: <b>${ev.lab_ndf_pct}%</b></div>
            <div style="color:${ev.ndf_pct_error <= 10 ? "#34d399" : "#fbbf24"};font-weight:700;">Error: ${ev.ndf_pct_error}%</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function switchGtChartMetric(metric) {
  currentGtChartMetric = metric;
  ["DM", "CP", "NDF"].forEach((k) => {
    const btn = document.getElementById(`btnGtMetric${k}`);
    if (btn) btn.classList.toggle("active", k === metric);
  });
  if (farmData && farmData.ground_truth_validation) {
    renderGtChart(farmData.ground_truth_validation.samples);
  }
}
window.switchGtChartMetric = switchGtChartMetric;

function filterGtTable(group) {
  currentGtSourceFilter = group;
  const mapBtn = {
    ALL: "btnGtFilterAll",
    FERNANDES_2024: "btnGtFilterFern",
    DLD_THAILAND: "btnGtFilterDld",
  };
  Object.entries(mapBtn).forEach(([k, id]) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("active", k === group);
  });
  if (farmData && farmData.ground_truth_validation) {
    renderGtSamplesTable(farmData.ground_truth_validation.samples);
  }
}
window.filterGtTable = filterGtTable;

function renderGtChart(samples) {
  const ctxEl = document.getElementById("gtComparisonChart");
  if (!ctxEl) return;
  const ctx = ctxEl.getContext("2d");
  if (gtComparisonChartInstance) {
    gtComparisonChartInstance.destroy();
  }

  const labels = samples.map((s) => `${s.sample_id.replace("BRAZIL-", "BR-").replace("TH-PAKCHONG-", "TH-")} (r${s.row10},c${s.col10})`);
  let labData = [];
  let predData = [];
  let unitTitle = "kg DM / ha";

  if (currentGtChartMetric === "DM") {
    labData = samples.map((s) => s.lab_dm_kg_ha);
    predData = samples.map((s) => s.pred_dm_kg_ha);
    unitTitle = "Green DM (kg DM/ha)";
  } else if (currentGtChartMetric === "CP") {
    labData = samples.map((s) => s.lab_cp_pct);
    predData = samples.map((s) => s.pred_cp_pct);
    unitTitle = "Crude Protein (% DM)";
  } else {
    labData = samples.map((s) => s.lab_ndf_pct);
    predData = samples.map((s) => s.pred_ndf_pct);
    unitTitle = "Neutral Detergent Fiber (% NDF)";
  }

  gtComparisonChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: `ค่าตรวจห้องแล็บจริง (Lab Ground Truth — ${unitTitle})`,
          data: labData,
          backgroundColor: "rgba(16, 185, 129, 0.78)",
          borderRadius: 5,
        },
        {
          label: `ค่าที่เว็บคำนวณจากพิกเซล .TIF (Web Predicted — ${unitTitle})`,
          data: predData,
          backgroundColor: "rgba(6, 182, 212, 0.78)",
          borderRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11.5 } } },
      },
      scales: {
        x: { ticks: { color: "#cbd5e1", font: { size: 10.2 } }, grid: { color: "rgba(148,163,184,0.06)" } },
        y: {
          beginAtZero: currentGtChartMetric === "DM",
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.08)" },
          title: { display: true, text: unitTitle, color: "#94a3b8" },
        },
      },
    },
  });
}

function renderGtSamplesTable(samples) {
  const tbody = document.getElementById("gtSamplesTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const filtered =
    currentGtSourceFilter === "ALL"
      ? samples
      : samples.filter((s) => s.source_group === currentGtSourceFilter);

  filtered.forEach((s) => {
    const tr = document.createElement("tr");
    const isFern = s.source_group === "FERNANDES_2024";
    tr.innerHTML = `
      <td style="font-family:'JetBrains Mono',monospace;">
        <b>${s.sample_id}</b><br/>
        <span class="th-sub">${s.date} • ${s.species_label.split(" ")[0]} (${s.rest_days}d)</span>
      </td>
      <td style="font-family:'JetBrains Mono',monospace;">
        <span style="background:${isFern ? "rgba(6,182,212,0.18);color:#38bdf8" : "rgba(16,185,129,0.18);color:#34d399"};padding:2px 7px;border-radius:99px;font-size:10.3px;font-weight:700;">
          ${s.scene_id}
        </span><br/>
        <span class="th-sub">Pixel: <b>row=${s.row10}, col=${s.col10}</b> (Lat ${s.lat}, Lon ${s.lon})</span>
      </td>
      <td style="font-family:'JetBrains Mono',monospace;">
        B4_dn: <b style="color:#f87171">${s.B4_dn}</b> | B8_dn: <b style="color:#34d399">${s.B8_dn}</b><br/>
        <a href="${s.cog_urls.B4}" target="_blank" style="color:#38bdf8;font-size:10.3px;text-decoration:underline;">[B04.tif]</a>
        <a href="${s.cog_urls.B8}" target="_blank" style="color:#38bdf8;font-size:10.3px;text-decoration:underline;margin-left:4px;">[B08.tif]</a>
      </td>
      <td style="font-family:'JetBrains Mono',monospace;">
        NDVI: <b style="color:#a3e635">${s.NDVI.toFixed(4)}</b><br/>
        NDRE: <b style="color:#38bdf8">${s.NDRE.toFixed(4)}</b>
      </td>
      <td style="font-family:'JetBrains Mono',monospace;">
        แล็บ: <b>${s.lab_dm_kg_ha.toLocaleString()}</b><br/>
        เว็บ: <b style="color:#38bdf8">${s.pred_dm_kg_ha.toLocaleString()}</b> kg/ha
      </td>
      <td style="font-family:'JetBrains Mono',monospace;color:${s.dm_mape_pct <= 14 ? "#34d399" : "#fbbf24"};font-weight:700;">
        ${s.dm_error_kg_ha >= 0 ? "+" : ""}${s.dm_error_kg_ha} kg/ha<br/>
        (${s.dm_mape_pct}%)
      </td>
      <td style="font-family:'JetBrains Mono',monospace;">
        แล็บ: <b>${s.lab_cp_pct}%</b><br/>
        เว็บ: <b style="color:#38bdf8">${s.pred_cp_pct}%</b>
      </td>
      <td style="font-family:'JetBrains Mono',monospace;color:${s.cp_mape_pct <= 12 ? "#34d399" : "#fbbf24"};font-weight:700;">
        ${s.cp_error_pct >= 0 ? "+" : ""}${s.cp_error_pct}%<br/>
        (${s.cp_mape_pct}%)
      </td>
      <td style="font-family:'JetBrains Mono',monospace;">
        แล็บ: <b>${s.lab_ndf_pct}%</b><br/>
        เว็บ: <b style="color:#38bdf8">${s.pred_ndf_pct}%</b> (${s.ndf_mape_pct}%)
      </td>
      <td>
        <button type="button" class="btn-chip active" onclick="inspectSamplePixelOnMap('${s.sample_id}')" style="font-size:11px;padding:4px 8px;">
          📍 ส่องพิกเซลจริง
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function runCustomGtLabCheck() {
  const pSel = document.getElementById("gtEvalPaddockSelect");
  const pid = (pSel && pSel.value) || "PADDOCK-A1";
  const labDm = parseFloat(document.getElementById("gtInputLabDm").value || "3480");
  const labCp = parseFloat(document.getElementById("gtInputLabCp").value || "15.9");
  const labNdf = parseFloat(document.getElementById("gtInputLabNdf").value || "60.4");

  const resp = await fetch("/api/validation/check-lab", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paddock_id: pid,
      lab_dm_kg_ha: labDm,
      lab_cp_pct: labCp,
      lab_ndf_pct: labNdf,
    }),
  });
  const data = await resp.json();
  const ev = data.evaluation;

  const box = document.getElementById("gtCustomEvalResult");
  if (box) {
    box.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
        <b>ผลการเทียบค่าตรวจแล็บจริงกับแปลง: <span style="color:#38bdf8">${data.paddock_name}</span></b>
        <span style="background:${ev.passed_trl4_standard ? "#10b981" : "#f59e0b"};color:#fff;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700;">
          ความแม่นยำรวม: ${ev.overall_accuracy_pct}% • ${ev.verdict_badge}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-family:'JetBrains Mono',monospace;font-size:11.8px;">
        <div style="background:rgba(15,23,42,0.5);padding:8px;border-radius:8px;">
          <div style="color:#94a3b8;font-size:10.5px;">มวลหญ้าแห้ง (Green DM)</div>
          <div>แล็บ: <b>${ev.lab_dm_kg_ha.toLocaleString()}</b> | เว็บ: <b style="color:#38bdf8">${ev.pred_dm_kg_ha.toLocaleString()}</b></div>
          <div style="color:${ev.dm_pct_error <= 15 ? "#34d399" : "#fbbf24"};font-weight:700;">Error: ±${ev.dm_abs_error} kg/ha (${ev.dm_pct_error}%)</div>
        </div>
        <div style="background:rgba(15,23,42,0.5);padding:8px;border-radius:8px;">
          <div style="color:#94a3b8;font-size:10.5px;">โปรตีนหยาบ (%CP Kjeldahl)</div>
          <div>แล็บ: <b>${ev.lab_cp_pct}%</b> | เว็บ: <b style="color:#38bdf8">${ev.pred_cp_pct}%</b></div>
          <div style="color:${ev.cp_pct_error <= 12 ? "#34d399" : "#fbbf24"};font-weight:700;">Error: ±${ev.cp_abs_error}% (${ev.cp_pct_error}%)</div>
        </div>
        <div style="background:rgba(15,23,42,0.5);padding:8px;border-radius:8px;">
          <div style="color:#94a3b8;font-size:10.5px;">เยื่อใยอาหาร (%NDF Van Soest)</div>
          <div>แล็บ: <b>${ev.lab_ndf_pct}%</b> | เว็บ: <b style="color:#38bdf8">${ev.pred_ndf_pct}%</b></div>
          <div style="color:${ev.ndf_pct_error <= 10 ? "#34d399" : "#fbbf24"};font-weight:700;">Error: ±${ev.ndf_abs_error}% (${ev.ndf_pct_error}%)</div>
        </div>
      </div>
    `;
  }
}



