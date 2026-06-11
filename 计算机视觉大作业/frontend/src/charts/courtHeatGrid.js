export function createCourtHeatGrid(container, insightElement, onSeek, bus) {
  const width = 640;
  const height = 270;
  const margin = { top: 18, right: 18, bottom: 28, left: 42 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const cols = 8;
  const rows = 12;
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);
  let state = null;
  let mode = "both";

  function validPlayers(data) {
    return (data.players || []).filter((row) => {
      if (mode !== "both" && row.role !== mode) return false;
      return row.court_x_m !== null && row.court_y_m !== null;
    });
  }

  function buildCells(data) {
    const courtW = data.analysis.court_size_m?.width || 6.1;
    const courtH = data.analysis.court_size_m?.length || 13.4;
    const cells = d3.range(rows * cols).map((idx) => ({
      row: Math.floor(idx / cols),
      col: idx % cols,
      count: 0,
      near: 0,
      far: 0,
      representative: null,
    }));
    for (const player of validPlayers(data)) {
      const col = Math.max(0, Math.min(cols - 1, Math.floor((player.court_x_m / courtW) * cols)));
      const row = Math.max(0, Math.min(rows - 1, Math.floor((player.court_y_m / courtH) * rows)));
      const cell = cells[row * cols + col];
      cell.count += 1;
      if (player.role === "near") cell.near += 1;
      if (player.role === "far") cell.far += 1;
      if (cell.representative === null) cell.representative = player.frame;
    }
    const maxCount = d3.max(cells, (cell) => cell.count) || 1;
    for (const cell of cells) {
      cell.ratio = cell.count / maxCount;
    }
    return cells;
  }

  function drawCourtLines() {
    const cellW = innerW / cols;
    const cellH = innerH / rows;
    g.selectAll(".heat-grid-line-x")
      .data(d3.range(cols + 1))
      .join("line")
      .attr("class", "heat-grid-line-x")
      .attr("x1", (d) => d * cellW)
      .attr("x2", (d) => d * cellW)
      .attr("y1", 0)
      .attr("y2", innerH);
    g.selectAll(".heat-grid-line-y")
      .data(d3.range(rows + 1))
      .join("line")
      .attr("class", "heat-grid-line-y")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", (d) => d * cellH)
      .attr("y2", (d) => d * cellH);
    g.selectAll(".heat-net")
      .data([rows / 2])
      .join("line")
      .attr("class", "heat-net")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", (d) => d * cellH)
      .attr("y2", (d) => d * cellH);
  }

  function updateInsight(cells) {
    if (!insightElement) return;
    const top = d3.greatest(cells, (cell) => cell.count);
    const label = mode === "both" ? "Both players" : mode === "near" ? "Near player" : "Far player";
    if (!top || top.count === 0) {
      insightElement.textContent = `${label}: no valid court-projected player rows.`;
      return;
    }
    insightElement.textContent = `${label}: hottest cell r${top.row + 1}/c${top.col + 1}, ${top.count} tracked frames. Click a cell to freeze a representative frame.`;
  }

  function updateData(data) {
    state = data;
    g.selectAll("*").remove();
    const cells = buildCells(data);
    const cellW = innerW / cols;
    const cellH = innerH / rows;

    g.append("rect")
      .attr("class", "heat-court-bg")
      .attr("width", innerW)
      .attr("height", innerH);

    g.selectAll(".heat-cell")
      .data(cells)
      .join("rect")
      .attr("class", "heat-cell")
      .attr("x", (cell) => cell.col * cellW)
      .attr("y", (cell) => cell.row * cellH)
      .attr("width", cellW)
      .attr("height", cellH)
      .attr("fill", (cell) => cell.count ? color(0.18 + cell.ratio * 0.82) : "#fafbfc")
      .on("click", (event, cell) => {
        if (cell.representative !== null) {
          onSeek(cell.representative, { pause: true });
          if (bus) bus.emit("select", { frame: cell.representative });
        }
      })
      .append("title")
      .text((cell) => `row ${cell.row + 1}, col ${cell.col + 1}: ${cell.count} frames`);

    drawCourtLines();
    g.append("text")
      .attr("class", "chart-caption")
      .attr("x", 0)
      .attr("y", innerH + 20)
      .text("8 x 12 court grid · darker cells mean more valid player frames");
    updateInsight(cells);
    renderFrame(0);
  }

  function renderFrame(frame) {
    if (!state) return;
    const courtW = state.analysis.court_size_m?.width || 6.1;
    const courtH = state.analysis.court_size_m?.length || 13.4;
    const active = validPlayers(state)
      .filter((row) => row.frame === frame)
      .map((row) => {
        const col = Math.max(0, Math.min(cols - 1, Math.floor((row.court_x_m / courtW) * cols)));
        const cellRow = Math.max(0, Math.min(rows - 1, Math.floor((row.court_y_m / courtH) * rows)));
        return `${cellRow}:${col}`;
      });
    const activeSet = new Set(active);
    g.selectAll(".heat-cell")
      .classed("active-frame", (cell) => activeSet.has(`${cell.row}:${cell.col}`));
  }

  function setMode(nextMode) {
    mode = nextMode;
    if (state) updateData(state);
  }

  return { updateData, renderFrame, setMode };
}
