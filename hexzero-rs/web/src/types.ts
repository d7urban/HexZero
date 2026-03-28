/** Move objects crossing the JS ↔ WASM boundary. */
export type HexMove =
  | { type: "cell"; row: number; col: number }
  | { type: "swap" };

/** Messages sent TO the AI worker. */
export interface SearchRequest {
  kind: "search";
  size: number;
  pieRule: boolean;
  moves: HexMove[];
  sims: number;
  useNN: boolean;
}

export interface EvalResponse {
  kind: "evalResponse";
  policy: Float32Array;
  value: number;
}

/** Messages sent FROM the AI worker. */
export interface SearchResult {
  kind: "result";
  move: HexMove;
}

export interface NeedEval {
  kind: "needEval";
  features: Float32Array;
  sizeNorm: number;
  planes: number;
  boardSize: number;
}

export type WorkerInbound = SearchRequest | EvalResponse;
export type WorkerOutbound = SearchResult | NeedEval;

/** Constants matching Rust hex-game. */
export const BLACK = 1;
export const WHITE = -1;
export const EMPTY = 0;
