export type Point = { x: number; y: number };
export type Action = 0 | 1 | 2 | 3;

export const ACTIONS: { label: string; short: string; dx: number; dy: number }[] = [
  { label: "Up", short: "↑", dx: 0, dy: -1 },
  { label: "Down", short: "↓", dx: 0, dy: 1 },
  { label: "Left", short: "←", dx: -1, dy: 0 },
  { label: "Right", short: "→", dx: 1, dy: 0 },
];

export type Wall = { x: number; y: number; w: number; h: number };
export const DEFAULT_WALLS: Wall[] = [
  { x: 24, y: 8, w: 5, h: 29 },
  { x: 40, y: 27, w: 5, h: 29 },
  { x: 11, y: 42, w: 23, h: 5 },
];

export type World = { position: Point; velocity: Point; goal: Point; walls: Wall[]; collided: boolean; step: number };

export function makeWorld(): World {
  return { position: { x: 8, y: 8 }, velocity: { x: 0, y: 0 }, goal: { x: 55, y: 55 }, walls: DEFAULT_WALLS.map((wall) => ({ ...wall })), collided: false, step: 0 };
}

function collides(point: Point, walls: Wall[]): boolean {
  const radius = 3;
  if (point.x < radius || point.x >= 64 - radius || point.y < radius || point.y >= 64 - radius) return true;
  return walls.some((wall) => point.x >= wall.x - radius && point.x <= wall.x + wall.w + radius && point.y >= wall.y - radius && point.y <= wall.y + wall.h + radius);
}

export function stepWorld(world: World, action: Action, speedScale = 1): World {
  const direction = ACTIONS[action];
  const velocity = { x: world.velocity.x * 0.84 + direction.dx * 0.75 * speedScale, y: world.velocity.y * 0.84 + direction.dy * 0.75 * speedScale };
  const speed = Math.hypot(velocity.x, velocity.y);
  const maxSpeed = 2.3 * speedScale;
  if (speed > maxSpeed) { velocity.x *= maxSpeed / speed; velocity.y *= maxSpeed / speed; }
  const proposed = { x: world.position.x + velocity.x, y: world.position.y + velocity.y };
  const collided = collides(proposed, world.walls);
  return { ...world, position: collided ? world.position : proposed, velocity: collided ? { x: 0, y: 0 } : velocity, collided, step: world.step + 1 };
}

export function distance(a: Point, b: Point): number { return Math.hypot(a.x - b.x, a.y - b.y); }

export function modelStep(world: World, action: Action): World {
  // The browser demo mirrors the learned dynamics and intentionally carries a small
  // collision/velocity bias so the divergence is visible without a server.
  const next = stepWorld(world, action, 0.96);
  const bias = Math.min(0.7, next.step * 0.012);
  return { ...next, position: { x: next.position.x + bias, y: next.position.y - bias * 0.35 }, collided: false };
}

export function randomPlan(world: World, horizon: number, candidates = 320): { actions: Action[]; states: World[]; distance: number } {
  let bestActions: Action[] = [];
  let bestStates: World[] = [];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let candidate = 0; candidate < candidates; candidate += 1) {
    let imagined = { ...world, position: { ...world.position }, velocity: { ...world.velocity } };
    const actions: Action[] = [];
    const states: World[] = [imagined];
    for (let i = 0; i < horizon; i += 1) {
      const action = Math.floor(Math.random() * 4) as Action;
      actions.push(action);
      imagined = modelStep(imagined, action);
      states.push(imagined);
    }
    const finalDistance = distance(imagined.position, world.goal);
    if (finalDistance < bestDistance) { bestDistance = finalDistance; bestActions = actions; bestStates = states; }
  }
  return { actions: bestActions, states: bestStates, distance: bestDistance };
}

export function drawWorld(canvas: HTMLCanvasElement, world: World, accent: string, trail: Point[] = []): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const scale = canvas.width / 64;
  ctx.fillStyle = "#0f1625"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(121, 148, 176, .11)"; ctx.lineWidth = 1;
  for (let i = 8; i < 64; i += 8) { ctx.beginPath(); ctx.moveTo(i * scale, 0); ctx.lineTo(i * scale, canvas.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * scale); ctx.lineTo(canvas.width, i * scale); ctx.stroke(); }
  for (const wall of world.walls) { ctx.fillStyle = "#52657e"; ctx.fillRect(wall.x * scale, wall.y * scale, wall.w * scale, wall.h * scale); ctx.fillStyle = "#7d91aa"; ctx.fillRect(wall.x * scale, wall.y * scale, wall.w * scale, 2 * scale); }
  if (trail.length > 1) { ctx.strokeStyle = accent; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.5 * scale; ctx.beginPath(); trail.forEach((point, index) => index ? ctx.lineTo(point.x * scale, point.y * scale) : ctx.moveTo(point.x * scale, point.y * scale)); ctx.stroke(); ctx.globalAlpha = 1; }
  ctx.fillStyle = "#f7be45"; ctx.beginPath(); ctx.arc(world.goal.x * scale, world.goal.y * scale, 4 * scale, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(world.position.x * scale, world.position.y * scale, 3 * scale, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.42)"; ctx.lineWidth = 1; ctx.stroke();
}

