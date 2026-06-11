function fmt(value, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function statusText(ball) {
  if (!ball) return "No row";
  if (ball.is_missing) return "Missing";
  if (ball.is_interpolated) return "Interpolated";
  if (ball.confidence !== null && ball.confidence < 0.5) return "Low confidence";
  if (!ball.is_spatial_valid) return "Image-only";
  return "Valid";
}

function rowAt(rows, frame, role = null) {
  return rows.find((row) => row.frame === frame && (!role || row.role === role)) || null;
}

function distanceToBall(ball, player) {
  if (!ball || !player || !ball.is_spatial_valid) return null;
  if (ball.court_x_m === null || ball.court_y_m === null || player.court_x_m === null || player.court_y_m === null) return null;
  return Math.hypot(ball.court_x_m - player.court_x_m, ball.court_y_m - player.court_y_m);
}

function playerBlock(label, player, motion, distance) {
  return `
    <div class="frame-detail-card">
      <span>${label}</span>
      <strong>${player ? `ID ${player.track_id ?? "n/a"}` : "missing"}</strong>
      <dl>
        <dt>Court</dt><dd>${player ? `${fmt(player.court_x_m)}m, ${fmt(player.court_y_m)}m` : "n/a"}</dd>
        <dt>Speed</dt><dd>${motion ? fmt(motion.speed_mps, 2, " m/s") : "n/a"}</dd>
        <dt>Distance</dt><dd>${motion ? fmt(motion.cumulative_distance_m, 1, " m") : "n/a"}</dd>
        <dt>Confidence</dt><dd>${player ? fmt(player.confidence, 3) : "n/a"}</dd>
        <dt>Near ball</dt><dd>${distance === null ? "n/a" : fmt(distance, 2, " m")}</dd>
      </dl>
    </div>
  `;
}

export function renderFrameDetail(container, data, frame) {
  if (!container || !data) return;
  const fps = data.analysis?.fps || 30;
  const ball = rowAt(data.ball || [], frame);
  const near = rowAt(data.players || [], frame, "near");
  const far = rowAt(data.players || [], frame, "far");
  const nearMotion = rowAt(data.motion || [], frame, "near");
  const farMotion = rowAt(data.motion || [], frame, "far");
  const nearDistance = distanceToBall(ball, near);
  const farDistance = distanceToBall(ball, far);
  const nearest = [
    { label: "Near", distance: nearDistance },
    { label: "Far", distance: farDistance },
  ].filter((item) => item.distance !== null).sort((a, b) => a.distance - b.distance)[0];

  container.innerHTML = `
    <div class="frame-detail-grid">
      <div class="frame-detail-card frame-detail-wide">
        <span>Frozen frame</span>
        <strong>${frame} · ${(frame / Math.max(fps, 1)).toFixed(3)}s</strong>
        <dl>
          <dt>Shuttle status</dt><dd>${statusText(ball)}</dd>
          <dt>Source</dt><dd>${ball?.source || "n/a"}</dd>
          <dt>Confidence</dt><dd>${ball ? fmt(ball.confidence, 3) : "n/a"}</dd>
          <dt>Pixel</dt><dd>${ball && ball.x_px !== null ? `${fmt(ball.x_px, 1)}, ${fmt(ball.y_px, 1)}` : "n/a"}</dd>
          <dt>Court</dt><dd>${ball && ball.is_spatial_valid ? `${fmt(ball.court_x_m)}m, ${fmt(ball.court_y_m)}m` : "not projected"}</dd>
          <dt>Ball speed</dt><dd>${ball?.speed_valid ? fmt(ball.speed_mps, 2, " m/s") : "not trusted"}</dd>
        </dl>
      </div>
      ${playerBlock("Near player", near, nearMotion, nearDistance)}
      ${playerBlock("Far player", far, farMotion, farDistance)}
      <div class="frame-detail-card frame-detail-wide">
        <span>Closest tracked player</span>
        <strong>${nearest ? `${nearest.label} · ${fmt(nearest.distance, 2, " m")}` : "n/a"}</strong>
        <p>Distance is a proximity cue only; it is not a hit detection or tactical label.</p>
      </div>
    </div>
  `;
}
