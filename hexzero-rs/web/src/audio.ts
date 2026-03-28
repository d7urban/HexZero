/**
 * Web Audio API chimes for game events.
 * Mirrors the desktop paplay-based chimes from hex-gui/app.rs.
 */

let audioCtx: AudioContext | null = null;

function getCtx(): AudioContext {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }
  return audioCtx;
}

function playChime(freq: number, duration: number): void {
  const ctx = getCtx();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.3, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
  osc.connect(gain).connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + duration);
}

export function playSwapChime(): void {
  playChime(880, 0.4);
}

export function playWinChime(): void {
  playChime(1047, 0.8);
}
