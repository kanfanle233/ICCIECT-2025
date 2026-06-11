import math
import random
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field


WIDTH = 900
HEIGHT = 720

HEX_RADIUS = 245.0
BALL_RADIUS = 11.0
MIN_BALL_RADIUS = 5.5

GRAVITY = 820.0
ANGULAR_SPEED = 0.85
RESTITUTION = 0.88
WALL_FRICTION = 0.025

OPEN_EDGE_INDEX = 1
MAX_TOTAL_GENERATED = 600
MAX_LIVE_BALLS = 280

FIXED_DT = 1.0 / 180.0
MAX_FRAME_DT = 1.0 / 30.0
OUTSIDE_MARGIN = 180.0

BACKGROUND = "#10141d"
WALL_COLOR = "#7dd3fc"
WALL_SHADOW = "#1d4f64"
OPEN_MARKER = "#fbbf24"
TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#9ca3af"

BALL_COLORS = [
    "#fb7185",
    "#f97316",
    "#facc15",
    "#34d399",
    "#38bdf8",
    "#a78bfa",
    "#f472b6",
]


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = BALL_RADIUS
    color: str = BALL_COLORS[0]
    escaped: bool = False
    cooldown: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=16))


class RotatingHexagonGame:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Rotating Hexagon Split Balls")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=WIDTH,
            height=HEIGHT,
            bg=BACKGROUND,
            highlightthickness=0,
        )
        self.canvas.pack()

        self.center = (WIDTH / 2.0, HEIGHT / 2.0 - 35.0)
        self.angle = 0.0
        self.balls: list[Ball] = []
        self.total_generated = 0
        self.game_over = False
        self.last_time = time.perf_counter()

        self.root.bind("<Key-r>", self.restart)
        self.root.bind("<Key-R>", self.restart)
        self.root.bind("<space>", self.restart)
        self.root.bind("<Button-1>", self.restart_after_game_over)
        self.root.focus_set()

        self.reset()

    def reset(self) -> None:
        self.angle = 0.0
        self.game_over = False
        self.total_generated = 1
        self.balls = [
            Ball(
                self.center[0],
                self.center[1] - 72.0,
                145.0,
                -80.0,
                BALL_RADIUS,
                BALL_COLORS[0],
            )
        ]
        self.last_time = time.perf_counter()

    def restart(self, _event: object | None = None) -> None:
        self.reset()

    def restart_after_game_over(self, _event: object | None = None) -> None:
        if self.game_over:
            self.reset()

    def run(self) -> None:
        self.tick()
        self.root.mainloop()

    def tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - self.last_time, MAX_FRAME_DT)
        self.last_time = now

        if not self.game_over:
            steps = max(1, math.ceil(dt / FIXED_DT))
            step_dt = dt / steps
            for _ in range(steps):
                self.step(step_dt)

        self.draw()
        self.root.after(16, self.tick)

    def step(self, dt: float) -> None:
        self.angle += ANGULAR_SPEED * dt
        vertices = self.hex_vertices()
        inward_normals = self.edge_inward_normals(vertices)
        new_balls: list[Ball] = []

        for ball in self.balls:
            ball.cooldown = max(0.0, ball.cooldown - dt)
            ball.vy += GRAVITY * dt
            ball.x += ball.vx * dt
            ball.y += ball.vy * dt
            ball.trail.append((ball.x, ball.y))

            if not ball.escaped:
                if self.crossed_open_edge(ball, vertices, inward_normals):
                    ball.escaped = True
                else:
                    self.resolve_wall_collisions(
                        ball,
                        vertices,
                        inward_normals,
                        new_balls,
                    )

        kept: list[Ball] = []
        for ball in self.balls:
            if not self.is_far_outside(ball):
                kept.append(ball)
        kept.extend(new_balls)
        self.balls = kept

        if not self.balls:
            self.game_over = True

    def hex_vertices(self) -> list[tuple[float, float]]:
        cx, cy = self.center
        vertices = []
        for i in range(6):
            a = self.angle + i * math.tau / 6.0
            vertices.append((cx + math.cos(a) * HEX_RADIUS, cy + math.sin(a) * HEX_RADIUS))
        return vertices

    @staticmethod
    def edge_inward_normals(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
        normals = []
        for i in range(6):
            ax, ay = vertices[i]
            bx, by = vertices[(i + 1) % 6]
            ex = bx - ax
            ey = by - ay
            length = math.hypot(ex, ey)
            normals.append((-ey / length, ex / length))
        return normals

    def crossed_open_edge(
        self,
        ball: Ball,
        vertices: list[tuple[float, float]],
        inward_normals: list[tuple[float, float]],
    ) -> bool:
        ax, ay = vertices[OPEN_EDGE_INDEX]
        nx, ny = inward_normals[OPEN_EDGE_INDEX]
        signed_distance = (ball.x - ax) * nx + (ball.y - ay) * ny
        return signed_distance < -ball.radius

    def resolve_wall_collisions(
        self,
        ball: Ball,
        vertices: list[tuple[float, float]],
        inward_normals: list[tuple[float, float]],
        new_balls: list[Ball],
    ) -> None:
        did_split = False
        for edge_index in range(6):
            if edge_index == OPEN_EDGE_INDEX:
                continue

            ax, ay = vertices[edge_index]
            bx, by = vertices[(edge_index + 1) % 6]
            ex = bx - ax
            ey = by - ay
            edge_len_sq = ex * ex + ey * ey
            if edge_len_sq == 0:
                continue

            t = ((ball.x - ax) * ex + (ball.y - ay) * ey) / edge_len_sq
            t = max(0.0, min(1.0, t))
            contact_x = ax + ex * t
            contact_y = ay + ey * t

            dx = ball.x - contact_x
            dy = ball.y - contact_y
            dist_sq = dx * dx + dy * dy
            if dist_sq >= ball.radius * ball.radius:
                continue

            dist = math.sqrt(max(dist_sq, 1e-12))
            if dist > 1e-6:
                nx = dx / dist
                ny = dy / dist
            else:
                nx, ny = inward_normals[edge_index]

            penetration = ball.radius - dist
            ball.x += nx * (penetration + 0.25)
            ball.y += ny * (penetration + 0.25)

            wall_vx, wall_vy = self.wall_velocity_at(contact_x, contact_y)
            rel_vx = ball.vx - wall_vx
            rel_vy = ball.vy - wall_vy
            normal_speed = rel_vx * nx + rel_vy * ny

            if normal_speed < 0.0:
                rel_vx -= (1.0 + RESTITUTION) * normal_speed * nx
                rel_vy -= (1.0 + RESTITUTION) * normal_speed * ny

                tx = -ny
                ty = nx
                tangent_speed = rel_vx * tx + rel_vy * ty
                rel_vx -= WALL_FRICTION * tangent_speed * tx
                rel_vy -= WALL_FRICTION * tangent_speed * ty

                ball.vx = rel_vx + wall_vx
                ball.vy = rel_vy + wall_vy

                if not did_split and self.can_split(new_balls) and ball.cooldown <= 0.0:
                    self.split_ball(ball, nx, ny, new_balls)
                    did_split = True

    def wall_velocity_at(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.center
        rx = x - cx
        ry = y - cy
        return (-ANGULAR_SPEED * ry, ANGULAR_SPEED * rx)

    def can_split(self, new_balls: list[Ball]) -> bool:
        return (
            self.total_generated < MAX_TOTAL_GENERATED
            and len(self.balls) + len(new_balls) < MAX_LIVE_BALLS
        )

    def split_ball(
        self,
        ball: Ball,
        nx: float,
        ny: float,
        new_balls: list[Ball],
    ) -> None:
        tx = -ny
        ty = nx
        speed = max(150.0, math.hypot(ball.vx, ball.vy))
        kick = min(180.0, 0.32 * speed)
        direction = 1.0 if random.random() < 0.5 else -1.0

        shared_vx = ball.vx + nx * 24.0
        shared_vy = ball.vy + ny * 24.0
        new_radius = max(MIN_BALL_RADIUS, ball.radius * 0.94)
        ball.radius = new_radius
        ball.vx = shared_vx + tx * kick * direction
        ball.vy = shared_vy + ty * kick * direction
        ball.cooldown = 0.08

        self.total_generated += 1
        child_color = BALL_COLORS[self.total_generated % len(BALL_COLORS)]
        child = Ball(
            ball.x - tx * new_radius * 1.4 * direction,
            ball.y - ty * new_radius * 1.4 * direction,
            shared_vx - tx * kick * direction,
            shared_vy - ty * kick * direction,
            new_radius,
            child_color,
            False,
            0.08,
        )
        child.trail.append((child.x, child.y))
        new_balls.append(child)

    @staticmethod
    def is_far_outside(ball: Ball) -> bool:
        return (
            ball.x < -OUTSIDE_MARGIN
            or ball.x > WIDTH + OUTSIDE_MARGIN
            or ball.y < -OUTSIDE_MARGIN
            or ball.y > HEIGHT + OUTSIDE_MARGIN
        )

    def draw(self) -> None:
        self.canvas.delete("frame")
        self.canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill=BACKGROUND, width=0, tags="frame")
        self.draw_hexagon()
        self.draw_balls()
        self.draw_hud()

    def draw_hexagon(self) -> None:
        vertices = self.hex_vertices()

        for edge_index in range(6):
            ax, ay = vertices[edge_index]
            bx, by = vertices[(edge_index + 1) % 6]
            if edge_index == OPEN_EDGE_INDEX:
                marker_radius = 5
                self.canvas.create_oval(
                    ax - marker_radius,
                    ay - marker_radius,
                    ax + marker_radius,
                    ay + marker_radius,
                    fill=OPEN_MARKER,
                    outline="",
                    tags="frame",
                )
                self.canvas.create_oval(
                    bx - marker_radius,
                    by - marker_radius,
                    bx + marker_radius,
                    by + marker_radius,
                    fill=OPEN_MARKER,
                    outline="",
                    tags="frame",
                )
                continue

            self.canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill=WALL_SHADOW,
                width=13,
                capstyle=tk.ROUND,
                tags="frame",
            )
            self.canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill=WALL_COLOR,
                width=7,
                capstyle=tk.ROUND,
                tags="frame",
            )

    def draw_balls(self) -> None:
        draw_trails = len(self.balls) <= 120
        if draw_trails:
            for ball in self.balls:
                if len(ball.trail) < 2:
                    continue
                points = []
                for x, y in ball.trail:
                    points.extend((x, y))
                self.canvas.create_line(
                    *points,
                    fill="#263244",
                    width=max(1, int(ball.radius / 3)),
                    smooth=True,
                    tags="frame",
                )

        for ball in self.balls:
            r = ball.radius
            self.canvas.create_oval(
                ball.x - r - 2,
                ball.y - r - 2,
                ball.x + r + 2,
                ball.y + r + 2,
                fill="#0b1020",
                outline="",
                tags="frame",
            )
            self.canvas.create_oval(
                ball.x - r,
                ball.y - r,
                ball.x + r,
                ball.y + r,
                fill=ball.color,
                outline="#f8fafc",
                width=1,
                tags="frame",
            )

    def draw_hud(self) -> None:
        if self.game_over:
            self.canvas.create_rectangle(
                0,
                0,
                WIDTH,
                HEIGHT,
                fill=BACKGROUND,
                stipple="gray50",
                width=0,
                tags="frame",
            )
            self.canvas.create_text(
                WIDTH / 2,
                HEIGHT / 2 - 28,
                anchor="center",
                fill=TEXT_COLOR,
                font=("Helvetica", 40, "bold"),
                text="GAME OVER",
                tags="frame",
            )
            self.canvas.create_text(
                WIDTH / 2,
                HEIGHT / 2 + 24,
                anchor="center",
                fill=MUTED_TEXT,
                font=("Helvetica", 18),
                text=f"Generated {self.total_generated} balls. Press R, Space, or click to restart.",
                tags="frame",
            )

        self.canvas.create_text(
            18,
            HEIGHT - 18,
            anchor="sw",
            fill=TEXT_COLOR,
            font=("Helvetica", 17, "bold"),
            text=f"Total balls: {self.total_generated}",
            tags="frame",
        )
        self.canvas.create_text(
            WIDTH - 18,
            HEIGHT - 18,
            anchor="se",
            fill=MUTED_TEXT,
            font=("Helvetica", 12),
            text="R / Space: restart",
            tags="frame",
        )


if __name__ == "__main__":
    RotatingHexagonGame().run()
