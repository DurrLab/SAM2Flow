#!/usr/bin/env python3
"""Quick sanity check for the SAM2Flow replica."""
import argparse
import torch

from configs.sam2flow_base import get_cfg
from core.Networks import build_network


def main():
    parser = argparse.ArgumentParser(description="SAM2Flow demo forward pass")
    parser.add_argument("--frames", type=int, default=2, help="number of frames to feed (>=2)")
    args = parser.parse_args()

    cfg = get_cfg()
    model = build_network(cfg)
    model.eval()

    height, width = cfg.image_size
    video = torch.randn(1, max(2, args.frames), 3, height, width)

    with torch.inference_mode():
        outputs = model(video, prompts=None, test_mode=True)

    flow = outputs["final"]
    print(f"Predicted flow tensor shape: {flow.shape}")
    print(f"Tracked ROI count (flattened): {flow.shape[0]}")


if __name__ == "__main__":
    main()
