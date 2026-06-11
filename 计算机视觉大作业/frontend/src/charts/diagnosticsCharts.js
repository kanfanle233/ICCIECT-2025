const QUALITY_METRICS = [
  { key: "score", label: "Score", value: (d) => d.quality.ball_quality_score, format: (v) => v.toFixed(2) },
  { key: "visible", label: "Visible", value: (d) => d.quality.ball_detection_rate * 100, format: (v) => `${v.toFixed(1)}%` },
  { key: "spatial", label: "Spatial", value: (d) => d.quality.ball_spatial_rate * 100, format: (v) => `${v.toFixed(1)}%` },
  { key: "gap", label: "Max gap", value: (d) => maxMissingGap(d.quality), format: (v) => `${Math.round(v)}f` },
  { key: "interp", label: "Interp", value: (d) => d.quality.ball_interpolated_frames || 0, format: (v) => `${Math.round(v)}f` },
  { key: "low", label: "Low conf", value: (d) => d.quality.ball_low_confidence_frames || 0, format: (v) => `${Math.round(v)}f` },
];

const SOURCE_KEYS = [
  "refined",
  "interp",
  "missing",
  "rejected_roi",
  "rejected_static_lock",
  "rejected_jump",
  "rejected_static_motion",
];

const SOURCE_COLORS = {
  refined: "#6aab8e",
  interp: "#e8945a",
  missing: "#c75555",
  rejected_roi: "#4a90d9",
  rejected_static_lock: "#7a6aad",
  rejected_jump: "#c75b7a",
  rejected_static_motion: "#888888",
  unknown: "#b0b0b0",
};

function datasetList(allDatasets) {
  if (!allDatasets) return [];
  return Array.from(allDatasets.values ? allDatasets.values() : allDatasets);
}

function maxMissingGap(quality) {
  const segments = quality?.ball_missing_segments || [];
  return d3.max(segments, (seg) => seg.length) || 0;
}

function titleOf(data) {
  return data.entry?.title || data.entry?.id || data.analysis?.video_id || "unknown";
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function ballCategory(row) {
  if (row.is_missing) return "missing";
  if (row.is_interpolated) return "interpolated";
  if (row.confidence !== null && row.confidence < 0.5) return "low";
  if (!row.is_spatial_valid) return "image-only";
  return "valid";
}

function categoryColor(category) {
  return {
    valid: "#6aab8e",
    "image-only": "#4a90d9",
    interpolated: "#e8945a",
    low: "#c9a84c",
    missing: "#c75555",
  }[category] || "#b0b0b0";
}

function shortLabel(label) {
  return label.replace("pro_match", "pm").replace("17_", "17-").replace("19_", "19-");
}

export function createQualityMatrix(container, onSelectVideo) {
  const width = 760;
  const height = 290;
  const margin = { top: 44, right: 18, bottom: 18, left: 128 };
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`);
  let sortKey = "score";
  let cache = [];
  let selectedId = null;

  function updateData(allDatasets, nextSelectedId) {
    cache = datasetList(allDatasets);
    selectedId = nextSelectedId;
    render();
  }

  function render() {
    svg.selectAll("*").remove();
    if (!cache.length) {
      svg.append("text").attr("class", "empty-note").attr("x", 20).attr("y", 40).text("Diagnostics unavailable");
      return;
    }
    const sortMetric = QUALITY_METRICS.find((metric) => metric.key === sortKey) || QUALITY_METRICS[0];
    const rows = [...cache].sort((a, b) => d3.descending(sortMetric.value(a), sortMetric.value(b)));
    const rowH = (height - margin.top - margin.bottom) / rows.length;
    const colW = (width - margin.left - margin.right) / QUALITY_METRICS.length;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    svg.append("text")
      .attr("class", "chart-caption")
      .attr("x", margin.left)
      .attr("y", 18)
      .text(`Click a metric header to sort · selected: ${selectedId || "none"}`);

    QUALITY_METRICS.forEach((metric, idx) => {
      const values = rows.map((row) => metric.value(row));
      const extent = d3.extent(values);
      const scale = d3.scaleLinear().domain(extent[0] === extent[1] ? [extent[0] - 1, extent[1] + 1] : extent).range([0.18, 1]);
      svg.append("text")
        .attr("class", `quality-matrix-head ${metric.key === sortKey ? "active" : ""}`)
        .attr("x", margin.left + idx * colW + colW / 2)
        .attr("y", 34)
        .attr("text-anchor", "middle")
        .text(metric.label)
        .on("click", () => {
          sortKey = metric.key;
          render();
        });
      g.selectAll(`.quality-cell-${metric.key}`)
        .data(rows)
        .join("rect")
        .attr("class", (row) => `quality-cell ${row.entry.id === selectedId ? "selected" : ""}`)
        .attr("x", idx * colW + 3)
        .attr("y", (_row, rowIdx) => rowIdx * rowH + 3)
        .attr("width", colW - 6)
        .attr("height", Math.max(12, rowH - 6))
        .attr("fill", (row) => d3.interpolateBlues(scale(metric.value(row))))
        .on("click", (_event, row) => onSelectVideo(row.entry.id, 0));
      g.selectAll(`.quality-value-${metric.key}`)
        .data(rows)
        .join("text")
        .attr("class", "quality-cell-value")
        .attr("x", idx * colW + colW / 2)
        .attr("y", (_row, rowIdx) => rowIdx * rowH + rowH / 2 + 3)
        .attr("text-anchor", "middle")
        .text((row) => metric.format(metric.value(row)));
    });

    g.selectAll(".quality-row-label")
      .data(rows)
      .join("text")
      .attr("class", (row) => `quality-row-label ${row.entry.id === selectedId ? "selected" : ""}`)
      .attr("x", -8)
      .attr("y", (_row, idx) => idx * rowH + rowH / 2 + 3)
      .attr("text-anchor", "end")
      .text((row) => shortLabel(titleOf(row)))
      .on("click", (_event, row) => onSelectVideo(row.entry.id, 0));
  }

  return { updateData };
}

export function createSourceStack(container, onSelectVideo) {
  const width = 760;
  const height = 290;
  const margin = { top: 24, right: 18, bottom: 48, left: 128 };
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`);

  function updateData(allDatasets, selectedId) {
    svg.selectAll("*").remove();
    const rows = datasetList(allDatasets);
    if (!rows.length) return;
    const data = rows.map((dataset) => {
      const counts = Object.fromEntries(SOURCE_KEYS.map((key) => [key, 0]));
      for (const row of dataset.ball || []) {
        const key = SOURCE_KEYS.includes(row.source) ? row.source : "unknown";
        counts[key] = (counts[key] || 0) + 1;
      }
      counts.unknown = counts.unknown || 0;
      return { id: dataset.entry.id, label: titleOf(dataset), total: dataset.ball.length, ...counts };
    });
    const keys = [...SOURCE_KEYS, "unknown"].filter((key) => data.some((row) => row[key] > 0));
    const y = d3.scaleBand().domain(data.map((row) => row.id)).range([margin.top, height - margin.bottom]).padding(0.22);
    const x = d3.scaleLinear().domain([0, d3.max(data, (row) => row.total) || 1]).range([margin.left, width - margin.right]);
    const stacked = d3.stack().keys(keys)(data);

    svg.append("g")
      .selectAll("g")
      .data(stacked)
      .join("g")
      .attr("fill", (series) => SOURCE_COLORS[series.key] || SOURCE_COLORS.unknown)
      .selectAll("rect")
      .data((series) => series.map((d) => ({ ...d, key: series.key })))
      .join("rect")
      .attr("class", (d) => `source-stack-bar ${d.data.id === selectedId ? "selected" : ""}`)
      .attr("x", (d) => x(d[0]))
      .attr("y", (d) => y(d.data.id))
      .attr("width", (d) => Math.max(0, x(d[1]) - x(d[0])))
      .attr("height", y.bandwidth())
      .on("click", (_event, d) => onSelectVideo(d.data.id, 0))
      .append("title")
      .text((d) => `${d.data.label} · ${d.key}: ${d.data[d.key]} frames`);

    svg.append("g")
      .selectAll("text")
      .data(data)
      .join("text")
      .attr("class", (row) => `quality-row-label ${row.id === selectedId ? "selected" : ""}`)
      .attr("x", margin.left - 8)
      .attr("y", (row) => y(row.id) + y.bandwidth() / 2 + 3)
      .attr("text-anchor", "end")
      .text((row) => shortLabel(row.label));

    svg.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(5));

    const legend = svg.append("g").attr("transform", `translate(${margin.left},${height - 22})`);
    keys.forEach((key, idx) => {
      const item = legend.append("g").attr("transform", `translate(${idx * 88},0)`);
      item.append("rect").attr("width", 8).attr("height", 8).attr("fill", SOURCE_COLORS[key] || SOURCE_COLORS.unknown);
      item.append("text").attr("class", "chart-caption").attr("x", 12).attr("y", 8).text(key.replace("rejected_", "rej_"));
    });
  }

  return { updateData };
}

export function createAllVideoTimeline(container, onSelectVideoFrame) {
  const width = 760;
  const height = 310;
  const margin = { top: 18, right: 16, bottom: 24, left: 128 };
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`);

  function updateData(allDatasets, selectedId) {
    svg.selectAll("*").remove();
    const datasets = datasetList(allDatasets);
    if (!datasets.length) return;
    const rowH = (height - margin.top - margin.bottom) / datasets.length;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const innerW = width - margin.left - margin.right;

    for (const [idx, dataset] of datasets.entries()) {
      const y = idx * rowH;
      const frameCount = dataset.analysis.frame_count || dataset.ball.length || 1;
      const x = d3.scaleLinear().domain([0, Math.max(frameCount - 1, 1)]).range([0, innerW]);
      const barW = Math.max(1.1, innerW / frameCount);
      g.append("text")
        .attr("class", `quality-row-label ${dataset.entry.id === selectedId ? "selected" : ""}`)
        .attr("x", -8)
        .attr("y", y + rowH / 2 + 3)
        .attr("text-anchor", "end")
        .text(shortLabel(titleOf(dataset)));
      g.selectAll(`.all-video-tick-${idx}`)
        .data(dataset.ball || [])
        .join("rect")
        .attr("class", "all-video-tick")
        .attr("x", (row) => x(row.frame))
        .attr("y", y + 4)
        .attr("width", barW)
        .attr("height", Math.max(7, rowH - 8))
        .attr("fill", (row) => categoryColor(ballCategory(row)))
        .on("click", (_event, row) => onSelectVideoFrame(dataset.entry.id, row.frame))
        .append("title")
        .text((row) => `${dataset.entry.id} · frame ${row.frame} · ${ballCategory(row)}`);
    }
    svg.append("text")
      .attr("class", "chart-caption")
      .attr("x", margin.left)
      .attr("y", height - 4)
      .text("Green valid · Blue image-only · Orange interpolated · Yellow low confidence · Red missing");
  }

  return { updateData };
}

export function createSpatialValidity(container) {
  const width = 760;
  const height = 310;
  const margin = { top: 22, right: 16, bottom: 28, left: 34 };
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`);
  let state = null;
  let currentFrame = 0;

  function updateData(data) {
    state = data;
    render();
  }

  function renderFrame(frame) {
    currentFrame = frame;
    renderMarkers();
  }

  function render() {
    svg.selectAll("*").remove();
    if (!state) return;
    const paneW = (width - margin.left - margin.right - 26) / 2;
    const paneH = height - margin.top - margin.bottom;
    const pixelW = state.analysis.original_video_meta?.width || 1280;
    const pixelH = state.analysis.original_video_meta?.height || 720;
    const courtW = state.analysis.court_size_m?.width || 6.1;
    const courtH = state.analysis.court_size_m?.length || 13.4;
    const px = d3.scaleLinear().domain([0, pixelW]).range([0, paneW]);
    const py = d3.scaleLinear().domain([0, pixelH]).range([0, paneH]);
    const cx = d3.scaleLinear().domain([0, courtW]).range([0, paneW]);
    const cy = d3.scaleLinear().domain([0, courtH]).range([0, paneH]);
    const left = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const right = svg.append("g").attr("transform", `translate(${margin.left + paneW + 26},${margin.top})`);

    for (const [group, title] of [[left, "Pixel detections"], [right, "Court-projected valid points"]]) {
      group.append("rect").attr("class", "diag-plot-bg").attr("width", paneW).attr("height", paneH);
      group.append("text").attr("class", "chart-caption").attr("x", 0).attr("y", -7).text(title);
    }
    right.append("line").attr("class", "heat-net").attr("x1", 0).attr("x2", paneW).attr("y1", cy(courtH / 2)).attr("y2", cy(courtH / 2));

    left.selectAll(".spatial-point-pixel")
      .data((state.ball || []).filter((row) => row.x_px !== null && row.y_px !== null))
      .join("circle")
      .attr("class", "spatial-point-pixel")
      .attr("cx", (row) => px(row.x_px))
      .attr("cy", (row) => py(row.y_px))
      .attr("r", 2.2)
      .attr("fill", (row) => categoryColor(ballCategory(row)));

    right.selectAll(".spatial-point-court")
      .data((state.ball || []).filter((row) => row.is_spatial_valid && row.court_x_m !== null && row.court_y_m !== null))
      .join("circle")
      .attr("class", "spatial-point-court")
      .attr("cx", (row) => cx(row.court_x_m))
      .attr("cy", (row) => cy(row.court_y_m))
      .attr("r", 2.4)
      .attr("fill", (row) => categoryColor(ballCategory(row)));
    renderMarkers();
  }

  function renderMarkers() {
    if (!state) return;
    const paneW = (width - margin.left - margin.right - 26) / 2;
    const paneH = height - margin.top - margin.bottom;
    const pixelW = state.analysis.original_video_meta?.width || 1280;
    const pixelH = state.analysis.original_video_meta?.height || 720;
    const courtW = state.analysis.court_size_m?.width || 6.1;
    const courtH = state.analysis.court_size_m?.length || 13.4;
    const px = d3.scaleLinear().domain([0, pixelW]).range([0, paneW]);
    const py = d3.scaleLinear().domain([0, pixelH]).range([0, paneH]);
    const cx = d3.scaleLinear().domain([0, courtW]).range([0, paneW]);
    const cy = d3.scaleLinear().domain([0, courtH]).range([0, paneH]);
    const row = (state.ball || []).find((item) => item.frame === currentFrame);
    svg.selectAll(".spatial-current").remove();
    if (!row || row.x_px === null || row.y_px === null) return;
    svg.append("circle")
      .attr("class", "spatial-current")
      .attr("cx", margin.left + px(row.x_px))
      .attr("cy", margin.top + py(row.y_px))
      .attr("r", 5);
    if (row.is_spatial_valid && row.court_x_m !== null && row.court_y_m !== null) {
      svg.append("circle")
        .attr("class", "spatial-current")
        .attr("cx", margin.left + paneW + 26 + cx(row.court_x_m))
        .attr("cy", margin.top + cy(row.court_y_m))
        .attr("r", 5);
    }
  }

  return { updateData, renderFrame };
}

export function createWorkloadPanel(container) {
  const width = 760;
  const height = 310;
  const margin = { top: 18, right: 18, bottom: 28, left: 44 };
  const innerW = width - margin.left - margin.right;
  const topH = 150;
  const bottomY = 182;
  const bottomH = 78;
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`);
  let state = null;
  let x = d3.scaleLinear().range([0, innerW]);

  function rowsFor(role) {
    return (state.motion || []).filter((row) => row.role === role);
  }

  function smoothRows(rows) {
    return rows.map((row, idx) => {
      const start = Math.max(0, idx - 3);
      const end = Math.min(rows.length, idx + 4);
      const avg = d3.mean(rows.slice(start, end), (item) => item.speed_mps || 0) || 0;
      return { ...row, rolling_speed_mps: avg };
    });
  }

  function updateData(data) {
    state = data;
    render();
  }

  function renderFrame(frame) {
    if (!state) return;
    svg.selectAll(".workload-cursor")
      .attr("x1", margin.left + x(frame))
      .attr("x2", margin.left + x(frame));
  }

  function render() {
    svg.selectAll("*").remove();
    if (!state) return;
    const near = smoothRows(rowsFor("near"));
    const far = smoothRows(rowsFor("far"));
    const frameCount = state.analysis.frame_count || 1;
    x = d3.scaleLinear().domain([0, Math.max(frameCount - 1, 1)]).range([0, innerW]);
    const maxSpeed = d3.max([...near, ...far], (row) => row.rolling_speed_mps) || 1;
    const ySpeed = d3.scaleLinear().domain([0, maxSpeed]).nice().range([topH, 0]);
    const diffRows = near.map((row, idx) => ({
      frame: row.frame,
      diff: (row.cumulative_distance_m || 0) - (far[idx]?.cumulative_distance_m || 0),
    }));
    const maxAbsDiff = d3.max(diffRows, (row) => Math.abs(row.diff)) || 1;
    const yDiff = d3.scaleLinear().domain([-maxAbsDiff, maxAbsDiff]).range([bottomY + bottomH, bottomY]);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const speedLine = d3.line().x((row) => x(row.frame)).y((row) => ySpeed(row.rolling_speed_mps));
    const diffArea = d3.area().x((row) => x(row.frame)).y0(yDiff(0) - margin.top).y1((row) => yDiff(row.diff) - margin.top);
    const diffLine = d3.line().x((row) => x(row.frame)).y((row) => yDiff(row.diff) - margin.top);

    g.append("g").attr("class", "axis").call(d3.axisLeft(ySpeed).ticks(4));
    g.append("path").datum(near).attr("class", "series speed-near").attr("d", speedLine);
    g.append("path").datum(far).attr("class", "series speed-far").attr("d", speedLine);
    g.append("text").attr("class", "chart-caption").attr("x", 0).attr("y", -4).text("Rolling speed (7-frame mean)");

    svg.append("path")
      .datum(diffRows)
      .attr("class", "workload-diff-area")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .attr("d", diffArea);
    svg.append("path")
      .datum(diffRows)
      .attr("class", "workload-diff-line")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .attr("d", diffLine);
    svg.append("line")
      .attr("class", "grid-line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", yDiff(0))
      .attr("y2", yDiff(0));
    svg.append("text")
      .attr("class", "chart-caption")
      .attr("x", margin.left)
      .attr("y", bottomY - 4)
      .text("Distance differential (near - far)");

    const peakRows = [
      ...near.sort((a, b) => d3.descending(a.rolling_speed_mps, b.rolling_speed_mps)).slice(0, 3).map((row) => ({ ...row, role: "near" })),
      ...far.sort((a, b) => d3.descending(a.rolling_speed_mps, b.rolling_speed_mps)).slice(0, 3).map((row) => ({ ...row, role: "far" })),
    ];
    g.selectAll(".workload-peak")
      .data(peakRows)
      .join("circle")
      .attr("class", (row) => `workload-peak ${row.role}`)
      .attr("cx", (row) => x(row.frame))
      .attr("cy", (row) => ySpeed(row.rolling_speed_mps))
      .attr("r", 3.5)
      .append("title")
      .text((row) => `${row.role} peak window · frame ${row.frame} · ${row.rolling_speed_mps.toFixed(2)} m/s`);

    svg.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(${margin.left},${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(6));
    svg.append("line")
      .attr("class", "workload-cursor cursor-line")
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom);
    renderFrame(0);
  }

  return { updateData, renderFrame };
}
