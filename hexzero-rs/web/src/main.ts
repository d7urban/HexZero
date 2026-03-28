/**
 * HexZero web UI — main thread.
 *
 * Manages game state (via hex-wasm WasmHexState on main thread),
 * renders the board on canvas, dispatches AI search to a Web Worker.
 * ONNX inference runs on the main thread; the worker requests evaluations
 * via postMessage when doing NN-backed MCTS.
 */

import init, { WasmHexState } from "../../crates/hex-wasm/pkg/hex_wasm.js";
import {
  fitLayout, drawBoard, nameForPlayer, makePlayerColors,
  type BoardLayout, type BoardState, type PlayerColors,
} from "./board-renderer";
import { playSwapChime, playWinChime } from "./audio";
import type { HexMove, SearchRequest, WorkerOutbound, EvalResponse } from "./types";
import { BLACK, WHITE } from "./types";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const canvas   = document.getElementById("board-canvas") as HTMLCanvasElement;
const ctx      = canvas.getContext("2d")!;
const statusBar = document.getElementById("status-bar")!;
const moveInfo  = document.getElementById("move-info")!;
const evalLabel = document.getElementById("eval-label")!;
const newGameBtn = document.getElementById("new-game-btn") as HTMLButtonElement;
const undoBtn    = document.getElementById("undo-btn") as HTMLButtonElement;
const swapBtn    = document.getElementById("swap-btn") as HTMLButtonElement;
const sizeGroup  = document.getElementById("size-group")!;
const blueType   = document.getElementById("blue-type") as HTMLSelectElement;
const redType    = document.getElementById("red-type") as HTMLSelectElement;
const simsSlider = document.getElementById("sims-slider") as HTMLInputElement;
const simsLabel  = document.getElementById("sims-label")!;
const pieCheck   = document.getElementById("pie-check") as HTMLInputElement;
const mirrorCheck = document.getElementById("mirror-check") as HTMLInputElement;
const rotateCheck = document.getElementById("rotate-check") as HTMLInputElement;
const firstPlayerSel = document.getElementById("first-player") as HTMLSelectElement;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const SIZES = [5, 7, 9, 11, 13];
let boardSize = 9;
let pieRule   = true;
let firstIsBlue = true;
let state: WasmHexState;
let layout: BoardLayout;
let colors: PlayerColors = makePlayerColors(true);
let aiWorker: Worker;
let aiSearching = false;
let nnLoaded = false;

// ONNX inference session (main thread)
let onnxSession: {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>>;
} | null = null;
let OrtTensor: new (type: string, data: Float32Array, dims: number[]) => unknown;

// Sims: log10 slider → actual value
function simsFromSlider(): number {
  return Math.round(10 ** parseFloat(simsSlider.value));
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

function createWorker(): Worker {
  const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  w.onmessage = onWorkerMessage;
  w.onerror = (e) => {
    console.error("[main] Worker error:", e);
  };
  return w;
}

async function main(): Promise<void> {
  await init();

  // Populate size buttons
  for (const s of SIZES) {
    const btn = document.createElement("button");
    btn.textContent = `${s}`;
    btn.dataset.size = `${s}`;
    btn.addEventListener("click", () => {
      boardSize = s;
      updateSizeButtons();
      newGame();
    });
    sizeGroup.appendChild(btn);
  }
  updateSizeButtons();

  // Start worker
  aiWorker = createWorker();

  // Event listeners
  newGameBtn.addEventListener("click", () => newGame());
  undoBtn.addEventListener("click", () => undo());
  swapBtn.addEventListener("click", () => doSwap());
  simsSlider.addEventListener("input", () => {
    simsLabel.textContent = `${simsFromSlider()}`;
  });
  pieCheck.addEventListener("change", () => {
    pieRule = pieCheck.checked;
    newGame();
  });
  mirrorCheck.addEventListener("change", () => {
    layout.mirrored = mirrorCheck.checked;
    render();
  });
  rotateCheck.addEventListener("change", () => {
    layout.transposed = rotateCheck.checked;
    render();
  });
  firstPlayerSel.addEventListener("change", () => {
    firstIsBlue = firstPlayerSel.value === "blue";
    colors = makePlayerColors(firstIsBlue);
    newGame();
  });
  canvas.addEventListener("click", onCanvasClick);
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
      e.preventDefault();
      undo();
    }
  });
  window.addEventListener("resize", () => resizeCanvas());

  // Init sims label
  simsLabel.textContent = `${simsFromSlider()}`;

  // Load ONNX model on main thread
  loadModel();

  // Start first game
  newGame();
}

// ---------------------------------------------------------------------------
// Neural net model loading (main thread)
// ---------------------------------------------------------------------------

async function loadModel(): Promise<void> {
  evalLabel.textContent = "Loading neural net...";
  try {
    const ort = await import("onnxruntime-web/wasm");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    // Point ORT at the Vite middleware that serves its WASM/JS from node_modules.
    ort.env.wasm.wasmPaths = "/ort/";

    OrtTensor = ort.Tensor as never;
    console.log("[main] ORT imported, creating inference session...");
    onnxSession = await ort.InferenceSession.create("/model.onnx", {
      executionProviders: ["wasm"],
    }) as typeof onnxSession;

    nnLoaded = true;
    evalLabel.textContent = "Neural net evaluator";
    console.log("[main] ONNX model loaded");
  } catch (err) {
    nnLoaded = false;
    onnxSession = null;
    const msg = err instanceof Error ? err.message : String(err);
    evalLabel.textContent = "Random evaluator";
    evalLabel.title = "NN load failed: " + msg;
    console.warn("[main] Failed to load ONNX model:", err);
  }
}

/** Run ONNX inference for a worker's evaluation request. */
async function handleEvalRequest(
  features: Float32Array,
  sizeNorm: number,
  planes: number,
  boardSize: number,
): Promise<void> {
  if (!onnxSession) return;

  const x = new OrtTensor("float32", features, [1, planes, boardSize, boardSize]);
  const sizeScalar = new OrtTensor("float32", new Float32Array([sizeNorm]), [1, 1]);

  const results = await onnxSession.run({ x, size_scalar: sizeScalar });

  const logPolicy = results["log_policy"].data;
  const policy = new Float32Array(logPolicy.length);
  for (let i = 0; i < logPolicy.length; i++) {
    policy[i] = Math.exp(logPolicy[i]);
  }
  const value = results["value"].data[0];

  const resp: EvalResponse = { kind: "evalResponse", policy, value };
  aiWorker.postMessage(resp);
}

// ---------------------------------------------------------------------------
// Game lifecycle
// ---------------------------------------------------------------------------

function newGame(): void {
  if (aiSearching) {
    aiWorker.terminate();
    aiWorker = createWorker();
    aiSearching = false;
  }
  state = new WasmHexState(boardSize, pieRule);
  resizeCanvas();
  render();
  maybeStartAiTurn();
}

function resizeCanvas(): void {
  const area = canvas.parentElement!;
  const dpr = window.devicePixelRatio || 1;
  const w = area.clientWidth;
  const h = area.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  layout = fitLayout(w, h, boardSize);
  // Preserve display toggles across resize
  layout.mirrored = mirrorCheck.checked;
  layout.transposed = rotateCheck.checked;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function buildBoardState(): BoardState {
  const lastMoveRaw = state.lastMove();
  let lastMove: HexMove | null = null;
  if (lastMoveRaw !== null && lastMoveRaw !== undefined) {
    lastMove = lastMoveRaw as HexMove;
  }

  const winPathArr: Array<{ type: string; row: number; col: number }> = state.winningPath() ?? [];
  const winSet = new Set<string>();
  for (const p of winPathArr) {
    winSet.add(`${p.row},${p.col}`);
  }

  return {
    size: boardSize,
    board: state.boardData(),
    lastMove,
    winningPath: winSet,
    currentPlayer: state.currentPlayer(),
    isTerminal: state.isTerminal(),
  };
}

function render(): void {
  const bs = buildBoardState();
  drawBoard(ctx, layout, bs, colors);
  updateStatusBar(bs);
  updateButtons(bs);
}

function updateStatusBar(bs: BoardState): void {
  if (bs.isTerminal) {
    const w = state.winner();
    const name = nameForPlayer(w, colors);
    statusBar.textContent = `${name} wins!`;
  } else {
    const name = nameForPlayer(bs.currentPlayer, colors);
    const isHuman = playerIsHuman(bs.currentPlayer);
    statusBar.textContent = `${name}'s turn${isHuman ? " (yours)" : " (AI thinking...)"}`;
  }

  const mc = state.moveCount();
  if (mc > 0) {
    const lm = state.lastMove();
    let moveStr = "";
    if (lm && lm.type === "cell") {
      moveStr = `${String.fromCharCode(65 + lm.col)}${lm.row + 1}`;
    } else if (lm && lm.type === "swap") {
      moveStr = "swap";
    }
    moveInfo.textContent = `Move ${mc}: ${moveStr}`;
  } else {
    moveInfo.textContent = "";
  }
}

function updateButtons(bs: BoardState): void {
  undoBtn.disabled = state.moveCount() === 0 || aiSearching;

  if (state.moveCount() === 1 && pieRule && !bs.isTerminal && playerIsHuman(bs.currentPlayer)) {
    swapBtn.style.display = "block";
  } else {
    swapBtn.style.display = "none";
  }
}

function updateSizeButtons(): void {
  for (const btn of sizeGroup.querySelectorAll("button")) {
    const b = btn as HTMLButtonElement;
    b.classList.toggle("active", b.dataset.size === `${boardSize}`);
  }
}

// ---------------------------------------------------------------------------
// Player type helpers
// ---------------------------------------------------------------------------

function playerIsHuman(player: number): boolean {
  if (player === BLACK) return blueType.value === "human";
  return redType.value === "human";
}

// ---------------------------------------------------------------------------
// Move application
// ---------------------------------------------------------------------------

function applyMove(move: HexMove): void {
  state.applyMove(move);

  if (move.type === "swap") {
    playSwapChime();
  }

  render();

  if (state.isTerminal()) {
    playWinChime();
    return;
  }

  maybeStartAiTurn();
}

function onCanvasClick(e: MouseEvent): void {
  if (aiSearching || state.isTerminal()) return;
  if (!playerIsHuman(state.currentPlayer())) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const cell = layout.posToCell(x, y);
  if (!cell) return;

  const move: HexMove = { type: "cell", row: cell.row, col: cell.col };
  if (!state.isLegal(move)) return;

  applyMove(move);
}

function doSwap(): void {
  if (aiSearching || state.isTerminal()) return;
  const move: HexMove = { type: "swap" };
  if (!state.isLegal(move)) return;
  applyMove(move);
}

// ---------------------------------------------------------------------------
// Undo
// ---------------------------------------------------------------------------

function undo(): void {
  if (state.moveCount() === 0 || aiSearching) return;

  const cp = state.currentPlayer();
  const isHvAI = playerIsHuman(cp) && !playerIsHuman(cp === BLACK ? WHITE : BLACK);
  const n = (isHvAI && state.moveCount() >= 2) ? 2 : 1;

  state.undo(n);
  render();
}

// ---------------------------------------------------------------------------
// AI
// ---------------------------------------------------------------------------

function maybeStartAiTurn(): void {
  if (state.isTerminal()) return;
  if (playerIsHuman(state.currentPlayer())) return;

  aiSearching = true;
  render();

  const req: SearchRequest = {
    kind: "search",
    size: boardSize,
    pieRule,
    moves: state.moveHistory(),
    sims: simsFromSlider(),
    useNN: nnLoaded,
  };
  aiWorker.postMessage(req);
}

function onWorkerMessage(e: MessageEvent<WorkerOutbound>): void {
  const msg = e.data;
  if (msg.kind === "result") {
    aiSearching = false;
    applyMove(msg.move);
  } else if (msg.kind === "needEval") {
    handleEvalRequest(msg.features, msg.sizeNorm, msg.planes, msg.boardSize);
  }
}

// ---------------------------------------------------------------------------
// Player type change → maybe trigger AI
// ---------------------------------------------------------------------------

blueType.addEventListener("change", () => {
  if (!state.isTerminal() && !aiSearching) maybeStartAiTurn();
});
redType.addEventListener("change", () => {
  if (!state.isTerminal() && !aiSearching) maybeStartAiTurn();
});

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

main();
