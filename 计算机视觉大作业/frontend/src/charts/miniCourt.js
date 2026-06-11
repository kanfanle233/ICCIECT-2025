export function createMiniCourt(container, onSeek, bus) {
  const root = d3.select(container);
  const svg = root.append("svg").attr("viewBox", "0 0 520 560").attr("role", "img");
  const padX = 34;
  const padY = 22;
  const courtW = 520 - padX * 2;
  const courtH = 560 - padY * 2;
  const x = d3.scaleLinear().range([padX, padX + courtW]);
  const y = d3.scaleLinear().range([padY, padY + courtH]);
  const court = svg.append("g");
  const trails = svg.append("g");
  const cursor = svg.append("g");
  const highlightG = svg.append("g").style("pointer-events", "none");
  const glyphG = svg.append("g").attr("transform", `translate(${520 - 4 * 12 - 10}, 10)`);

  const CELL = 10;
  const GAP = 2;
  const glyphCells = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 4; c++) {
      const cell = glyphG.append("rect")
        .attr("x", c * (CELL + GAP))
        .attr("y", r * (CELL + GAP))
        .attr("width", CELL).attr("height", CELL)
        .attr("rx", 1.5)
        .attr("fill", "#e8e8e8")
        .attr("opacity", 0.35);
      glyphCells.push(cell);
    }
  }

  const COLORS = {
    on: "#6aab8e",
    off: "#e8e8e8",
    amber: "#c9a84c",
    red: "#c75555",
    orange: "#e8945a",
    blue: "#4a90d9",
    rose: "#c75b7a",
    teal: "#5ba8a0",
    gray: "#cccccc",
  };

  function updateGlyph(frame) {
    if (!state) return;
    const ball = state.ball?.[frame];
    const near = state.players?.find(d => d.frame === frame && d.role === "near");
    const far = state.players?.find(d => d.frame === frame && d.role === "far");

    // Row 0: Ball | Near | Far | Source
    const ballOn = ball && !ball.is_missing;
    glyphCells[0].attr("fill", ballOn ? COLORS.on : COLORS.red).attr("opacity", ballOn ? 0.85 : 0.5);
    const nearOn = near && near.court_x_m !== null;
    glyphCells[1].attr("fill", nearOn ? COLORS.rose : COLORS.off).attr("opacity", nearOn ? 0.85 : 0.25);
    const farOn = far && far.court_x_m !== null;
    glyphCells[2].attr("fill", farOn ? COLORS.teal : COLORS.off).attr("opacity", farOn ? 0.85 : 0.25);

    let srcColor = COLORS.gray;
    if (ball) {
      if (ball.is_missing) srcColor = COLORS.red;
      else if (ball.is_interpolated) srcColor = COLORS.orange;
      else if (!ball.is_spatial_valid) srcColor = COLORS.blue;
      else srcColor = COLORS.on;
    }
    glyphCells[3].attr("fill", srcColor).attr("opacity", 0.75);

    // Row 1: Speed | Confidence | Gap | Interpolated
    const ballRow = ball;
    let speedColor = COLORS.gray;
    if (ballRow && ballRow.speed_mps !== null) {
      speedColor = ballRow.speed_mps < 8 ? COLORS.amber : ballRow.speed_mps < 15 ? COLORS.blue : COLORS.on;
    }
    glyphCells[4].attr("fill", speedColor).attr("opacity", 0.7);

    let confColor = COLORS.gray;
    if (ballRow && ballRow.confidence !== null) {
      confColor = ballRow.confidence < 0.5 ? COLORS.red : ballRow.confidence < 0.8 ? COLORS.amber : COLORS.on;
    }
    glyphCells[5].attr("fill", confColor).attr("opacity", 0.7);

    const inGap = ballRow && ballRow.is_missing;
    glyphCells[6].attr("fill", inGap ? COLORS.red : COLORS.off).attr("opacity", inGap ? 0.6 : 0.2);

    const isInterp = ballRow && ballRow.is_interpolated;
    glyphCells[7].attr("fill", isInterp ? COLORS.orange : COLORS.off).attr("opacity", isInterp ? 0.6 : 0.2);
  }

  let state = null;
  let courtMode = "trajectory";
  let currentFrameId = 0;
  let selectedFrame = null;

  if (bus) {
    bus.on("select", (d) => {
      selectedFrame = d.frame;
      highlightSelectedFrame();
    });
    bus.on("clearSelect", () => {
      selectedFrame = null;
      highlightG.selectAll("*").remove();
      trails.selectAll("path").attr("opacity", 1);
      trails.selectAll("circle").attr("opacity", (dd) => dd && dd[1] !== undefined ? 0.12 : 0.08);
    });
  }

  function highlightSelectedFrame() {
    if (!state || selectedFrame === null) return;
    highlightG.selectAll("*").remove();
    trails.selectAll("path").attr("opacity", 0.15);

    const ball = state.ball?.[selectedFrame];
    const near = state.players?.find(d => d.frame === selectedFrame && d.role === "near");
    const far = state.players?.find(d => d.frame === selectedFrame && d.role === "far");

    if (ball && ball.court_x_m !== null) {
      highlightG.append("circle")
        .attr("cx", x(ball.court_x_m)).attr("cy", y(ball.court_y_m))
        .attr("r", 8).attr("fill", "#e8945a").attr("opacity", 0.5);
    }
    if (near && near.court_x_m !== null) {
      highlightG.append("circle")
        .attr("cx", x(near.court_x_m)).attr("cy", y(near.court_y_m))
        .attr("r", 8).attr("fill", "#c75b7a").attr("opacity", 0.5);
    }
    if (far && far.court_x_m !== null) {
      highlightG.append("circle")
        .attr("cx", x(far.court_x_m)).attr("cy", y(far.court_y_m))
        .attr("r", 8).attr("fill", "#5ba8a0").attr("opacity", 0.5);
    }
  }

  const line = d3.line()
    .defined((d) => d.court_x_m !== null && d.court_y_m !== null)
    .x((d) => x(d.court_x_m))
    .y((d) => y(d.court_y_m));

  function drawCourt(analysis) {
    const width = analysis.court_size_m?.width || 6.1;
    const length = analysis.court_size_m?.length || 13.4;
    x.range([padX, padX + courtW]);
    x.domain([0, width]);
    y.domain([0, length]);
    court.selectAll("*").remove();
    court.append("rect")
      .attr("class", "court-bg")
      .attr("x", padX)
      .attr("y", padY)
      .attr("width", courtW)
      .attr("height", courtH);
      
    const singlesMargin = width * (0.46 / 6.1);
    const shortServiceFromNet = length * (1.98 / 13.4);
    const doublesServiceFromBaseline = length * (0.76 / 13.4);
    const netY = length / 2;
    const centerLineTopEndY = netY - shortServiceFromNet;
    const centerLineBottomStartY = netY + shortServiceFromNet;

    const lines = [
      // Outer Boundary (Doubles sideline and baseline)
      [[0, 0], [width, 0], [width, length], [0, length], [0, 0]],
      
      // Singles sidelines (Left and Right)
      [[singlesMargin, 0], [singlesMargin, length]],
      [[width - singlesMargin, 0], [width - singlesMargin, length]],
      
      // Net Line
      [[0, netY], [width, netY]],
      
      // Short Service Lines (Top and Bottom)
      [[0, centerLineTopEndY], [width, centerLineTopEndY]],
      [[0, centerLineBottomStartY], [width, centerLineBottomStartY]],
      
      // Doubles Long Service Lines (Top and Bottom)
      [[0, doublesServiceFromBaseline], [width, doublesServiceFromBaseline]],
      [[0, length - doublesServiceFromBaseline], [width, length - doublesServiceFromBaseline]],
      
      // Center Service Lines (Top and Bottom service courts only)
      [[width / 2, 0], [width / 2, centerLineTopEndY]],
      [[width / 2, centerLineBottomStartY], [width / 2, length]]
    ];

    for (const l of lines) {
      court
        .append("path")
        .attr("class", "court-line")
        .attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(l));
    }
  }

  function rowsFor(role) {
    if (!state) return [];
    return state.players
      .filter((d) => d.role === role && d.court_x_m !== null && d.court_y_m !== null)
      .sort((a, b) => a.frame - b.frame);
  }

  function renderDensity() {
    if (!state) return;
    trails.selectAll("*").remove();
    
    const allNear = rowsFor("near");
    const allFar = rowsFor("far");
    const allBall = state.ball.filter((d) => !d.is_missing && d.court_x_m !== null && d.court_y_m !== null);

    trails.selectAll(".density-near")
      .data(allNear)
      .join("circle")
      .attr("class", "density-near")
      .attr("r", 2.2)
      .attr("fill", "#c75b7a")
      .attr("opacity", 0.08)
      .attr("cx", (d) => x(d.court_x_m))
      .attr("cy", (d) => y(d.court_y_m));

    trails.selectAll(".density-far")
      .data(allFar)
      .join("circle")
      .attr("class", "density-far")
      .attr("r", 2.2)
      .attr("fill", "#5ba8a0")
      .attr("opacity", 0.08)
      .attr("cx", (d) => x(d.court_x_m))
      .attr("cy", (d) => y(d.court_y_m));

    trails.selectAll(".density-ball")
      .data(allBall)
      .join("circle")
      .attr("class", "density-ball")
      .attr("r", 1.8)
      .attr("fill", "#e8945a")
      .attr("opacity", 0.12)
      .attr("cx", (d) => x(d.court_x_m))
      .attr("cy", (d) => y(d.court_y_m));
  }

  function renderFrame(frame) {
    if (!state) return;
    currentFrameId = frame;
    cursor.selectAll("*").remove();
    
    const current = {
      near: rowsFor("near").findLast((d) => d.frame <= frame),
      far: rowsFor("far").findLast((d) => d.frame <= frame),
      ball: state.ball.filter((d) => !d.is_missing && d.court_x_m !== null && d.court_y_m !== null).findLast((d) => d.frame <= frame),
    };

    if (courtMode === "trajectory") {
      trails.selectAll("*").remove();
      
      const nearTrail = rowsFor("near").filter((d) => d.frame <= frame);
      const farTrail = rowsFor("far").filter((d) => d.frame <= frame);
      const ballTrail = state.ball.filter((d) => !d.is_missing && d.court_x_m !== null && d.court_y_m !== null && d.frame <= frame);

      trails.append("path").datum(nearTrail).attr("class", "trajectory-near").attr("d", line);
      trails.append("path").datum(farTrail).attr("class", "trajectory-far").attr("d", line);
      trails.append("path").datum(ballTrail).attr("class", "trajectory-ball").attr("d", line);
    }

    const points = [
      ["near", current.near, "#c75b7a", 5.5],
      ["far", current.far, "#5ba8a0", 5.5],
      ["ball", current.ball, "#e8945a", 4.5],
    ];

    cursor
      .selectAll("circle")
      .data(points.filter((d) => d[1]))
      .join("circle")
      .attr("class", "current-marker")
      .attr("r", (d) => d[3])
      .attr("fill", (d) => d[2])
      .attr("stroke", "#333333")
      .attr("stroke-width", 1.5)
      .attr("cx", (d) => x(d[1].court_x_m))
      .attr("cy", (d) => y(d[1].court_y_m));

    updateGlyph(frame);
    return current;
  }

  function updateData(data) {
    state = data;
    drawCourt(data.analysis);
    if (courtMode === "density") {
      renderDensity();
    } else {
      trails.selectAll("*").remove();
    }
    svg.on("click", (event) => {
      const [mx, my] = d3.pointer(event);
      const nearest = data.ball
        .filter((d) => !d.is_missing && d.court_x_m !== null && d.court_y_m !== null)
        .reduce((best, row) => {
          const dist = Math.hypot(x(row.court_x_m) - mx, y(row.court_y_m) - my);
          return !best || dist < best.dist ? { row, dist } : best;
        }, null);
      if (nearest) {
        onSeek(nearest.row.frame);
        if (bus) bus.emit("select", { frame: nearest.row.frame });
      }
    });
    renderFrame(currentFrameId);
  }

  function setCourtMode(nextMode) {
    courtMode = nextMode;
    if (!state) return;
    if (courtMode === "density") {
      renderDensity();
    } else {
      trails.selectAll("*").remove();
    }
    renderFrame(currentFrameId);
  }

  return { updateData, renderFrame, setCourtMode };
}
