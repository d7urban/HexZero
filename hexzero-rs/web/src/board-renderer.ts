/**
 * Hex board geometry and Canvas 2D rendering.
 *
 * Ported from crates/hex-gui/src/board.rs.
 *
 * Coordinate system — pointy-top hexagons in an offset grid:
 *   cx = origin.x + col·dx + row·dx/2
 *   cy = origin.y + row·dy
 * where dx = √3·r, dy = 1.5·r.
 *
 * Mirror — flips the board left-right by reflecting each cx around the
 * board's x-centre, so the acute angle moves from top-left to top-right.
 *
 * Transpose — swaps (row, col) before computing display position,
 * visually rotating the board 90°.
 */

import { BLACK, WHITE, EMPTY, type HexMove } from "./types";

const ROOT3 = Math.sqrt(3);

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

const BLUE  = "rgb(60,100,220)";
const RED   = "rgb(220,60,60)";
const GOLD  = "rgb(255,215,0)";
const CELL_EMPTY = "rgb(220,220,220)";
const STROKE_NORMAL = "rgb(60,60,60)";
const STROKE_LAST   = "rgb(200,200,200)";
const LABEL_COLOR   = "rgb(80,80,80)";

/** Colour scheme driven by firstIsBlue flag. */
export interface PlayerColors {
  blackFill: string;
  whiteFill: string;
  blackName: string;
  whiteName: string;
}

export function makePlayerColors(firstIsBlue: boolean): PlayerColors {
  return firstIsBlue
    ? { blackFill: BLUE, whiteFill: RED, blackName: "Blue", whiteName: "Red" }
    : { blackFill: RED, whiteFill: BLUE, blackName: "Red", whiteName: "Blue" };
}

export function colorForPlayer(player: number, colors: PlayerColors): string {
  return player === BLACK ? colors.blackFill : colors.whiteFill;
}

export function nameForPlayer(player: number, colors: PlayerColors): string {
  return player === BLACK ? colors.blackName : colors.whiteName;
}

// ---------------------------------------------------------------------------
// BoardLayout — geometry
// ---------------------------------------------------------------------------

export class BoardLayout {
  size: number;
  radius: number;
  originX: number;
  originY: number;
  mirrored = false;
  transposed = false;

  constructor(size: number, radius: number, originX: number, originY: number) {
    this.size = size;
    this.radius = radius;
    this.originX = originX;
    this.originY = originY;
  }

  /** Horizontal step between adjacent cells in the same row. */
  dx(): number { return ROOT3 * this.radius; }

  /** Vertical step between adjacent rows. */
  dy(): number { return 1.5 * this.radius; }

  /** X of rightmost cell centre (cell (n-1, n-1)). */
  private rawMaxX(): number {
    const n = this.size - 1;
    return this.originX + n * this.dx() + n * this.dx() * 0.5;
  }

  /** Mirror axis (board centre x). */
  boardMidX(): number {
    return (this.originX + this.rawMaxX()) * 0.5;
  }

  /** Cell centre in raw coordinates (no mirror/transpose). */
  rawCenter(row: number, col: number): [number, number] {
    const dx = this.dx();
    return [
      this.originX + col * dx + row * dx * 0.5,
      this.originY + row * this.dy(),
    ];
  }

  /** Apply mirror to a raw x coordinate. */
  private mirrorX(x: number): number {
    return 2 * this.boardMidX() - x;
  }

  /** Screen position of game cell (row, col), with transpose + mirror. */
  cellCenter(row: number, col: number): [number, number] {
    const [r, c] = this.transposed ? [col, row] : [row, col];
    const [x, y] = this.rawCenter(r, c);
    return this.mirrored ? [this.mirrorX(x), y] : [x, y];
  }

  /** Screen position in raw display grid (no transpose, mirror applied). */
  rawCellCenter(rawRow: number, rawCol: number): [number, number] {
    const [x, y] = this.rawCenter(rawRow, rawCol);
    return this.mirrored ? [this.mirrorX(x), y] : [x, y];
  }

  /** 6 vertices of pointy-top hex at given centre. */
  hexVertices(cx: number, cy: number): [number, number][] {
    const r = this.radius;
    const pts: [number, number][] = [];
    for (let i = 0; i < 6; i++) {
      const angle = Math.PI * (i / 3 - 0.5);
      pts.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
    }
    return pts;
  }

  /** 6 vertices at raw display position (no transpose, mirror applied). */
  rawHexVerticesAt(rawRow: number, rawCol: number): [number, number][] {
    return this.hexVertices(...this.rawCellCenter(rawRow, rawCol));
  }

  /** Map screen position to game cell, or null if off-board. */
  posToCell(px: number, py: number): { row: number; col: number } | null {
    // Undo mirror so we work in raw coordinates.
    if (this.mirrored) {
      px = this.mirrorX(px);
    }

    const dy = this.dy();
    const rowF = (py - this.originY) / dy;
    const rowLo = Math.max(0, Math.floor(rowF - 1.5));
    const rowHi = Math.min(this.size, Math.ceil(rowF + 1.5) + 1);

    let bestDist2 = Infinity;
    let bestR = -1, bestC = -1;

    for (let r = rowLo; r < rowHi; r++) {
      const dx = this.dx();
      const colF = (px - this.originX - r * dx * 0.5) / dx;
      const colLo = Math.max(0, Math.floor(colF - 1.5));
      const colHi = Math.min(this.size, Math.ceil(colF + 1.5) + 1);
      for (let c = colLo; c < colHi; c++) {
        const [cx, cy] = this.rawCenter(r, c);
        const d2 = (cx - px) ** 2 + (cy - py) ** 2;
        if (d2 < bestDist2) {
          bestDist2 = d2;
          bestR = r;
          bestC = c;
        }
      }
    }

    if (bestDist2 <= (this.radius * 1.05) ** 2) {
      // raw (bestR, bestC) → game cell: reverse transpose
      if (this.transposed) {
        return { row: bestC, col: bestR };
      }
      return { row: bestR, col: bestC };
    }
    return null;
  }

  /** X-span of a display row (raw coords, mirror applied). */
  rowXSpan(row: number): [number, number] {
    const halfDx = this.dx() * 0.5;
    let xMin = Infinity, xMax = -Infinity;
    for (let c = 0; c < this.size; c++) {
      const [x] = this.rawCellCenter(row, c);
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
    }
    return [xMin - halfDx, xMax + halfDx];
  }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

export interface BoardState {
  size: number;
  /** Row-major flat board: BLACK(1), WHITE(-1), EMPTY(0). */
  board: Int8Array;
  lastMove: HexMove | null;
  winningPath: Set<string>; // "row,col" keys
  currentPlayer: number;
  isTerminal: boolean;
}

/** Compute layout that fits the board in the available canvas space. */
export function fitLayout(
  canvasW: number,
  canvasH: number,
  size: number,
): BoardLayout {
  const pad = 40;
  const availW = canvasW - 2 * pad;
  const availH = canvasH - 2 * pad;

  const wFactor = ROOT3 * 1.5 * (size - 1) + 2;
  const hFactor = 1.5 * (size - 1) + 2;

  const r = Math.min(availW / wFactor, availH / hFactor);

  const boardW = r * wFactor;
  const boardH = r * hFactor;
  const ox = (canvasW - boardW) / 2 + r;
  const oy = (canvasH - boardH) / 2 + r;

  return new BoardLayout(size, r, ox, oy);
}

/** Draw the complete hex board. */
export function drawBoard(
  ctx: CanvasRenderingContext2D,
  layout: BoardLayout,
  state: BoardState,
  colors: PlayerColors,
): void {
  const { size, radius: r } = layout;

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Edge strips
  drawEdgeStrips(ctx, layout, colors);

  // Cells
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const [cx, cy] = layout.cellCenter(row, col);
      const verts = layout.hexVertices(cx, cy);
      const cellVal = state.board[row * size + col];
      const isLast =
        state.lastMove !== null &&
        state.lastMove.type === "cell" &&
        state.lastMove.row === row &&
        state.lastMove.col === col;
      const inWin = state.winningPath.has(`${row},${col}`);

      // Fill
      let fill: string;
      if (inWin) {
        fill = GOLD;
      } else if (cellVal === BLACK) {
        fill = colors.blackFill;
      } else if (cellVal === WHITE) {
        fill = colors.whiteFill;
      } else {
        fill = CELL_EMPTY;
      }

      // Draw hex polygon
      ctx.beginPath();
      ctx.moveTo(verts[0][0], verts[0][1]);
      for (let i = 1; i < 6; i++) {
        ctx.lineTo(verts[i][0], verts[i][1]);
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = isLast ? STROKE_LAST : STROKE_NORMAL;
      ctx.lineWidth = isLast ? 2.5 : 1.5;
      ctx.stroke();

      // Last-move white dot
      if (isLast && cellVal !== EMPTY) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * 0.18, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.47)";
        ctx.fill();
      }
    }
  }

  // Coordinate labels
  drawCoordLabels(ctx, layout);
}

/** Colored edge strips indicating each player's connection goal. */
function drawEdgeStrips(
  ctx: CanvasRenderingContext2D,
  layout: BoardLayout,
  colors: PlayerColors,
): void {
  const { size, radius: r } = layout;
  const strip = r * 0.65;

  // When transposed, BLACK connects left↔right and WHITE top↔bottom,
  // so the band colours swap.
  const topbotColor = layout.transposed ? colors.whiteFill : colors.blackFill;
  const sidesColor  = layout.transposed ? colors.blackFill : colors.whiteFill;

  // Top band (raw display coords)
  const [topXl, topXr] = layout.rowXSpan(0);
  const [, yTop0] = layout.rawCellCenter(0, 0);
  const yTop = yTop0 - r * 0.5;
  ctx.beginPath();
  ctx.moveTo(topXl, yTop);
  ctx.lineTo(topXr, yTop);
  ctx.lineTo(topXr, yTop - strip);
  ctx.lineTo(topXl, yTop - strip);
  ctx.closePath();
  ctx.fillStyle = topbotColor;
  ctx.fill();

  // Bottom band
  const [botXl, botXr] = layout.rowXSpan(size - 1);
  const [, yBot0] = layout.rawCellCenter(size - 1, 0);
  const yBot = yBot0 + r * 0.5;
  ctx.beginPath();
  ctx.moveTo(botXl, yBot);
  ctx.lineTo(botXr, yBot);
  ctx.lineTo(botXr, yBot + strip);
  ctx.lineTo(botXl, yBot + strip);
  ctx.closePath();
  ctx.fillStyle = topbotColor;
  ctx.fill();

  // Left and right diagonal bands
  // Determine which visual column is leftmost / rightmost (mirror-aware).
  const [x0] = layout.rawCellCenter(0, 0);
  const [xN] = layout.rawCellCenter(0, size - 1);
  const colLeft  = x0 < xN ? 0 : size - 1;
  const colRight = size - 1 - colLeft;

  drawSideBand(ctx, layout, colLeft,  true,  strip, sidesColor);
  drawSideBand(ctx, layout, colRight, false, strip, sidesColor);
}

/** Parallelogram band along one side column. */
function drawSideBand(
  ctx: CanvasRenderingContext2D,
  layout: BoardLayout,
  col: number,
  outwardLeft: boolean,
  strip: number,
  color: string,
): void {
  const { size, radius: r } = layout;

  const viTop = outwardLeft ? 5 : 1;
  const viBot = outwardLeft ? 4 : 2;

  const topVerts = layout.rawHexVerticesAt(0, col);
  const botVerts = layout.rawHexVerticesAt(size - 1, col);

  const topInner = topVerts[viTop];
  const botInner = botVerts[viBot];

  const edgeX = botInner[0] - topInner[0];
  const edgeY = botInner[1] - topInner[1];
  const len = Math.sqrt(edgeX * edgeX + edgeY * edgeY) || 1e-6;

  let normalX: number, normalY: number;
  if (outwardLeft) {
    normalX = -edgeY / len;
    normalY = edgeX / len;
  } else {
    normalX = edgeY / len;
    normalY = -edgeX / len;
  }

  const offX = normalX * strip;
  const offY = normalY * strip;

  const extra = r * 0.5;
  const extX = (edgeX / len) * extra;
  const extY = (edgeY / len) * extra;

  const tlI: [number, number] = [topInner[0] - extX, topInner[1] - extY];
  const blI: [number, number] = [botInner[0] + extX, botInner[1] + extY];
  const tlO: [number, number] = [tlI[0] + offX, tlI[1] + offY];
  const blO: [number, number] = [blI[0] + offX, blI[1] + offY];

  ctx.beginPath();
  ctx.moveTo(tlI[0], tlI[1]);
  ctx.lineTo(blI[0], blI[1]);
  ctx.lineTo(blO[0], blO[1]);
  ctx.lineTo(tlO[0], tlO[1]);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

/** Row/column coordinate labels (raw display coords, mirror-aware). */
function drawCoordLabels(
  ctx: CanvasRenderingContext2D,
  layout: BoardLayout,
): void {
  const { size, radius: r } = layout;
  const fontSize = r * 0.55;
  ctx.font = `${fontSize}px monospace`;
  ctx.fillStyle = LABEL_COLOR;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // When transposed, display rows = game columns (letters)
  // and display columns = game rows (numbers).
  for (let row = 0; row < size; row++) {
    const [cx, cy] = layout.rawCellCenter(row, 0);
    const offX = layout.mirrored ? r * 1.5 : -r * 1.5;
    const label = layout.transposed
      ? String.fromCharCode(65 + row)
      : `${row + 1}`;
    ctx.fillText(label, cx + offX, cy);
  }

  for (let col = 0; col < size; col++) {
    const [cx, cy] = layout.rawCellCenter(0, col);
    const label = layout.transposed
      ? `${col + 1}`
      : String.fromCharCode(65 + col);
    ctx.fillText(label, cx, cy - r * 1.5);
  }
}
