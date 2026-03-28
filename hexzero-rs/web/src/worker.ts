/**
 * Web Worker for MCTS AI search.
 *
 * Two modes:
 * - Uniform evaluator: mctsSearch (synchronous, self-contained)
 * - Neural net: MctsSession (iterative), requesting evaluation from main thread
 */

import init, { mctsSearch, MctsSession } from "../../crates/hex-wasm/pkg/hex_wasm.js";
import type { WorkerInbound, WorkerOutbound, NeedEval } from "./types";

let ready = false;

async function initialize(): Promise<void> {
  await init();
  ready = true;
}

const initPromise = initialize();

/** Promise resolver for pending eval responses from main thread. */
let evalResolve: ((msg: { policy: Float32Array; value: number }) => void) | null = null;

/** Request NN evaluation from main thread and wait for the response. */
function requestEval(
  features: Float32Array,
  sizeNorm: number,
  planes: number,
  boardSize: number,
): Promise<{ policy: Float32Array; value: number }> {
  const msg: NeedEval = { kind: "needEval", features, sizeNorm, planes, boardSize };
  self.postMessage(msg);
  return new Promise((resolve) => {
    evalResolve = resolve;
  });
}

/** Run MCTS with neural net evaluation via iterative MctsSession. */
async function searchWithNN(
  size: number,
  pieRule: boolean,
  moves: unknown,
  sims: number,
): Promise<unknown> {
  const session = new MctsSession(size, pieRule, moves, sims);

  while (!session.done()) {
    const leaf = session.selectLeaf();
    if (leaf !== null && leaf !== undefined) {
      const { policy, value } = await requestEval(
        leaf.features,
        leaf.sizeNorm,
        leaf.planes,
        leaf.boardSize,
      );
      session.supplyEval(policy, value);
    }
  }

  const best = session.bestMove();
  session.free();
  return best;
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  if (!ready) await initPromise;

  const msg = e.data;

  if (msg.kind === "evalResponse") {
    // Main thread returned NN evaluation — resolve the pending promise
    if (evalResolve) {
      evalResolve({ policy: msg.policy, value: msg.value });
      evalResolve = null;
    }
    return;
  }

  if (msg.kind === "search") {
    try {
      let bestMove;
      if (msg.useNN) {
        bestMove = await searchWithNN(msg.size, msg.pieRule, msg.moves, msg.sims);
      } else {
        bestMove = mctsSearch(msg.size, msg.pieRule, msg.moves, msg.sims);
      }
      const response: WorkerOutbound = { kind: "result", move: bestMove };
      self.postMessage(response);
    } catch (err) {
      console.error("[worker] Search failed:", err);
      const bestMove = mctsSearch(msg.size, msg.pieRule, msg.moves, msg.sims);
      const response: WorkerOutbound = { kind: "result", move: bestMove };
      self.postMessage(response);
    }
  }
};
