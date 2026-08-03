import { useEffect, useRef, useState } from "react";
import { loadPocketWorldModel, OnnxWorldModel } from "./onnx";
import { ACTIONS, Action, World, distance, drawWorld, makeWorld, modelStep, randomPlan, stepWorld } from "./sim";

const initialActions: Action[] = [3, 3, 1, 1, 3, 3, 1, 1];
type EditMode = "start" | "goal" | "wall";

function Canvas({ world, accent, trail, onEdit }: { world: World; accent: string; trail: { x: number; y: number }[]; onEdit?: (point: { x: number; y: number }) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => { if (ref.current) drawWorld(ref.current, world, accent, trail); }, [world, accent, trail]);
  return <canvas ref={ref} width={512} height={512} className="world-canvas" aria-label="2D world visualization" onPointerDown={(event) => {
    if (!onEdit || !ref.current) return;
    const bounds = ref.current.getBoundingClientRect();
    onEdit({ x: ((event.clientX - bounds.left) / bounds.width) * 64, y: ((event.clientY - bounds.top) / bounds.height) * 64 });
  }} />;
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="sparkline-empty">play a rollout to collect drift</div>;
  const max = Math.max(...values, 1);
  const points = values.map((value, index) => `${(index / (values.length - 1)) * 100},${96 - (value / max) * 84}`).join(" ");
  return <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="prediction error over time"><polyline points={points} /></svg>;
}

export default function App() {
  const [real, setReal] = useState<World>(makeWorld());
  const [imagined, setImagined] = useState<World>(makeWorld());
  const [actions, setActions] = useState<Action[]>(initialActions);
  const [cursor, setCursor] = useState(0);
  const [running, setRunning] = useState(false);
  const [plan, setPlan] = useState<Action[] | null>(null);
  const [planDistance, setPlanDistance] = useState<number | null>(null);
  const [horizon, setHorizon] = useState(10);
  const [message, setMessage] = useState("Ready to compare reality with imagination.");
  const [editMode, setEditMode] = useState<EditMode | null>(null);
  const [driftHistory, setDriftHistory] = useState<number[]>([0]);
  const [predictor, setPredictor] = useState<OnnxWorldModel | null>(null);
  const realTrail = useRef<{ x: number; y: number }[]>([]);
  const imaginedTrail = useRef<{ x: number; y: number }[]>([]);

  useEffect(() => { void loadPocketWorldModel().then(setPredictor); }, []);

  const reset = () => {
    const world = makeWorld();
    setReal(world); setImagined(world); setCursor(0); setPlan(null); setPlanDistance(null); setRunning(false); setDriftHistory([0]);
    realTrail.current = [world.position]; imaginedTrail.current = [world.position];
    setMessage("Environment reset. Choose an action sequence or run a plan.");
  };

  const editMap = (point: { x: number; y: number }) => {
    if (!editMode) return;
    const next = { ...real, position: { ...real.position }, goal: { ...real.goal }, walls: real.walls.map((wall) => ({ ...wall })) };
    if (editMode === "start") next.position = { x: Math.max(4, Math.min(60, point.x)), y: Math.max(4, Math.min(60, point.y)) };
    if (editMode === "goal") next.goal = { x: Math.max(4, Math.min(60, point.x)), y: Math.max(4, Math.min(60, point.y)) };
    if (editMode === "wall") next.walls.push({ x: Math.max(2, Math.min(57, point.x - 2.5)), y: Math.max(2, Math.min(57, point.y - 2.5)), w: 5, h: 5 });
    setReal(next); setImagined({ ...next, position: { ...next.position }, goal: { ...next.goal }, walls: next.walls.map((wall) => ({ ...wall })) });
    setMessage(`${editMode} updated. Press Reset to restore the default map.`);
  };

  const play = async (sequence = actions) => {
    if (!sequence.length || running) return;
    setRunning(true); setCursor(0); setPlan(null); setPlanDistance(null);
    let realState = { ...makeWorld(), position: { ...real.position }, goal: { ...real.goal }, walls: real.walls.map((wall) => ({ ...wall })) };
    let imaginedState = { ...realState, position: { ...realState.position }, velocity: { ...realState.velocity } };
    realTrail.current = [realState.position]; imaginedTrail.current = [imaginedState.position]; setDriftHistory([0]); setReal(realState); setImagined(imaginedState);
    try {
      for (let index = 0; index < sequence.length; index += 1) {
        const action = sequence[index];
        const nextReal = stepWorld(realState, action);
        let nextImagined = modelStep(imaginedState, action);
        if (predictor) {
          const predictedPosition = await predictor.predictPosition(imaginedState, action);
          if (predictedPosition) nextImagined = { ...imaginedState, position: predictedPosition, velocity: { x: 0, y: 0 }, collided: false, step: imaginedState.step + 1 };
        }
        realState = nextReal; imaginedState = nextImagined;
        realTrail.current = [...realTrail.current, nextReal.position]; imaginedTrail.current = [...imaginedTrail.current, nextImagined.position];
        setReal(nextReal); setImagined(nextImagined); setCursor(index + 1); setDriftHistory((current) => [...current, distance(nextReal.position, nextImagined.position)]);
        await new Promise((resolve) => window.setTimeout(resolve, 140));
      }
      setMessage(predictor ? "ONNX rollout complete. Compare the learned prediction with ground truth." : "Rollout complete. Browser fallback is active until an ONNX checkpoint is present.");
    } finally { setRunning(false); }
  };

  const runPlan = () => {
    const result = randomPlan(real, horizon);
    setPlan(result.actions); setPlanDistance(result.distance);
    setMessage(`Random shooting searched 320 imagined futures. Best imagined distance: ${result.distance.toFixed(1)} px.`);
    void play(result.actions);
  };
  const addAction = (action: Action) => { if (actions.length < 32) setActions((current) => [...current, action]); };
  const removeLast = () => setActions((current) => current.slice(0, -1));
  const delta = distance(real.position, imagined.position);
  const realGoalDistance = distance(real.position, real.goal);
  const imaginedGoalDistance = distance(imagined.position, imagined.goal);

  return <main className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">◈</span><span>PocketWorld</span><span className="beta">LAB / 0.2</span></div><a href="https://github.com/ZhouYinLong-lab/Pocket-World-Model" target="_blank" rel="noreferrer">GitHub ↗</a></header>
    <section className="hero"><div><p className="eyebrow">OBSERVABLE WORLD MODEL LABORATORY</p><h1>How far can a tiny model <em>imagine?</em></h1><p className="lede">A small neural world model learns the motion rules of a 2D world, predicts the future, and plans through its own imagination.</p></div><div className="question-card"><span className="card-label">RESEARCH QUESTION</span><strong>When does prediction error break planning?</strong><span className="card-line"><i /> deterministic 64 × 64 environment</span></div></section>
    <section className="map-tools"><span className="eyebrow">MAP EDITOR</span><button className={editMode === "start" ? "selected" : ""} onClick={() => setEditMode(editMode === "start" ? null : "start")}>set start</button><button className={editMode === "goal" ? "selected" : ""} onClick={() => setEditMode(editMode === "goal" ? null : "goal")}>set goal</button><button className={editMode === "wall" ? "selected" : ""} onClick={() => setEditMode(editMode === "wall" ? null : "wall")}>draw wall</button><span className="map-help">{editMode ? `click the real simulator to ${editMode}` : "edit the start, goal, or walls before rollout"}</span><span className={`model-status ${predictor ? "ready" : "fallback"}`}><i />{predictor ? "ONNX MODEL READY" : "BROWSER FALLBACK"}</span></section>
    <section className="workspace"><div className="panel real-panel"><div className="panel-head"><div><span className="status-dot real-dot" />REAL SIMULATOR</div><span className="pill">GROUND TRUTH</span></div><Canvas world={real} accent="#5de0b7" trail={realTrail.current} onEdit={editMap} /><div className="panel-foot"><span>step <b>{real.step}</b></span><span>goal dist. <b>{realGoalDistance.toFixed(1)} px</b></span>{real.collided && <span className="collision">collision</span>}</div></div><div className="bridge"><span>vs</span><div className="bridge-line" /></div><div className="panel model-panel"><div className="panel-head"><div><span className="status-dot model-dot" />MODEL IMAGINATION</div><span className="pill accent-pill">{predictor ? "ONNX" : "PREDICTED"}</span></div><Canvas world={imagined} accent="#bb8cff" trail={imaginedTrail.current} /><div className="panel-foot"><span>step <b>{imagined.step}</b></span><span>goal dist. <b>{imaginedGoalDistance.toFixed(1)} px</b></span><span className="drift">drift <b>{delta.toFixed(1)} px</b></span></div></div></section>
    <section className="control-grid"><div className="control-card sequence-card"><div className="section-head"><div><span className="eyebrow">01 / ROLLOUT</span><h2>Feed it an action sequence</h2></div><span className="step-count">{actions.length} STEPS</span></div><div className="action-sequence">{actions.map((action, index) => <button key={`${index}-${action}`} className={`action-chip ${index < cursor ? "played" : ""}`} onClick={() => setActions((current) => current.filter((_, itemIndex) => itemIndex !== index))} title="Remove action">{ACTIONS[action].short}</button>)}{actions.length === 0 && <span className="empty-sequence">Add actions below</span>}</div><div className="action-buttons">{ACTIONS.map((action, index) => <button key={action.label} className="direction-button" onClick={() => addAction(index as Action)}><span>{action.short}</span>{action.label}</button>)}<button className="ghost-button" onClick={removeLast}>undo</button><button className="primary-button" onClick={() => void play()} disabled={running || actions.length === 0}>{running ? "Playing…" : "Play both futures"}<span>↗</span></button></div></div><div className="control-card planner-card"><div className="section-head"><div><span className="eyebrow">02 / PLANNING</span><h2>Search in imagination</h2></div><span className="status-live"><i /> RANDOM SHOOTING</span></div><p>Sample hundreds of futures in the learned model. Execute the best sequence in both worlds.</p><label className="range-label">PLANNING HORIZON <b>{horizon} STEPS</b></label><input type="range" min="4" max="32" value={horizon} onChange={(event) => setHorizon(Number(event.target.value))} /><button className="plan-button" onClick={runPlan} disabled={running}>Find a path <span>✦</span></button>{plan && <div className="plan-result"><span>best imagined distance</span><strong>{planDistance?.toFixed(1)} px</strong><small>{plan.map((action) => ACTIONS[action].short).join(" ")}</small></div>}</div></section>
    <section className="metrics"><div className="metrics-intro"><span className="eyebrow">LIVE READOUT</span><h2>Prediction error is a time series.</h2><p>Every step is another chance for a tiny mistake to compound.</p><Sparkline values={driftHistory} /></div><div className="metric"><span>POSITION DRIFT</span><strong>{delta.toFixed(1)}<small> px</small></strong><div className="meter"><i style={{ width: `${Math.min(100, delta * 10)}%` }} /></div></div><div className="metric"><span>REAL SUCCESS</span><strong className={realGoalDistance <= 4 ? "success" : ""}>{realGoalDistance <= 4 ? "YES" : "—"}</strong><p>goal radius: 4 px</p></div><div className="metric"><span>MODEL SUCCESS</span><strong className={imaginedGoalDistance <= 4 ? "success" : ""}>{imaginedGoalDistance <= 4 ? "YES" : "—"}</strong><p>model confidence: {Math.max(0, 100 - delta * 12).toFixed(0)}%</p></div></section>
    <footer><span>POCKETWORLD / A TINY WORLD MODEL EXPERIMENT</span><span>{message}</span></footer>
  </main>;
}
