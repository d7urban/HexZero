#!/usr/bin/env python3
"""
Export a HexZero checkpoint to ONNX for browser-side inference.

Usage:
    python export_onnx.py <checkpoint.pt> [output.onnx] [--size 11]

The exported model accepts dynamic board sizes (H, W) so one ONNX file
works for all board sizes. Use --size to set the example size for tracing
(default 11).

Inputs:
  x:            (B, 9, H, W) float32 — feature planes
  size_scalar:  (B, 1) float32       — normalised board size

Outputs:
  log_policy:   (B, H*W+1) float32   — log-softmax over moves
  value:        (B,) float32          — tanh value in [-1, 1]
"""

import sys
import os
import argparse

import torch

# Add project root to path so we can import config / net
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HexZeroConfig
from hexzero.net import HexNet


def export(src_path: str, dst_path: str, example_size: int = 11) -> None:
    cfg = HexZeroConfig()
    net = HexNet(cfg).cpu()
    net.eval()

    # Load checkpoint (.safetensors or .pt)
    print(f"Loading  {src_path}")
    if src_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(src_path)
    else:
        ckpt = torch.load(src_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

    # Strip torch.compile prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("_orig_mod.")] = v
    net.load_state_dict(cleaned, strict=False)

    # Example inputs for tracing
    B = 1
    H = W = example_size
    x = torch.randn(B, cfg.num_input_planes, H, W)
    size_scalar = torch.tensor([[0.5]])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)

    # Export using the legacy TorchScript-based exporter (dynamo=False)
    print(f"Exporting to {dst_path} (example size {example_size}x{example_size})")
    torch.onnx.export(
        net,
        (x, size_scalar),
        dst_path,
        input_names=["x", "size_scalar"],
        output_names=["log_policy", "value"],
        dynamic_axes={
            "x": {0: "batch", 2: "height", 3: "width"},
            "size_scalar": {0: "batch"},
            "log_policy": {0: "batch", 1: "moves"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"Saved    {dst_path}  ({size_mb:.1f} MB)")

    # Verify with ONNX Runtime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(dst_path)
        ort_out = sess.run(None, {
            "x": x.numpy(),
            "size_scalar": size_scalar.numpy(),
        })
        pt_out = net(x, size_scalar)
        lp_err = (torch.tensor(ort_out[0]) - pt_out[0]).abs().max().item()
        v_err  = (torch.tensor(ort_out[1]) - pt_out[1]).abs().max().item()
        print(f"Verification: log_policy max err={lp_err:.6f}, value max err={v_err:.6f}")
        if lp_err < 1e-4 and v_err < 1e-4:
            print("  PASS")
        else:
            print("  WARNING: larger than expected deviation")
    except ImportError:
        print("(onnxruntime not installed, skipping verification)")


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(project_root, "hexzero-rs", "checkpoints", "best.safetensors")
    default_out  = os.path.join(project_root, "hexzero-rs", "web", "public", "model.onnx")

    parser = argparse.ArgumentParser(description="Export HexZero to ONNX")
    parser.add_argument("checkpoint", nargs="?", default=default_ckpt,
                        help="Path to checkpoint (default: hexzero-rs/checkpoints/best.safetensors)")
    parser.add_argument("output", nargs="?", help="Output .onnx path (default: web/public/model.onnx)")
    parser.add_argument("--size", type=int, default=11, help="Example board size for tracing")
    args = parser.parse_args()

    dst = args.output if args.output else default_out

    export(args.checkpoint, dst, args.size)


if __name__ == "__main__":
    main()
