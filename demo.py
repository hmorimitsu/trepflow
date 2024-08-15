from argparse import ArgumentParser

import cv2 as cv
import numpy as np
import torch

from trepflow.trepflow import TrepFlow
from trepflow.trepflow_vars import models_dict
from flow_utils import flow_to_rgb


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="trepflow_l",
        choices=tuple(models_dict.keys()),
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs=2,
        default=(
            "sample_data/sintel_alley_1_frame_0001.png",
            "sample_data/sintel_alley_1_frame_0002.png",
        ),
        help=("Paths to the two input images."),
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    return parser


@torch.no_grad()
def infer(args, model):
    model.eval()

    images = [cv.imread(args.input_path[i]) for i in range(2)]
    assert len(images) == 2

    print("Loading images:")
    for p in args.input_path:
        print(f"- {p}")

    images = torch.from_numpy(np.stack(images, 0))
    images = images.float().permute(0, 3, 1, 2)[None] / 255.0

    if torch.cuda.is_available():
        model = model.cuda()
        images = images.cuda()
        if args.fp16:
            model = model.half()
            images = images.half()

    preds = model({"images": images})
    flow = preds["flows"][0, 0].permute(1, 2, 0).detach().cpu().numpy()
    flow_rgb = flow_to_rgb(flow)
    output_path = "flow_prediction.png"
    cv.imwrite(output_path, cv.cvtColor(flow_rgb, cv.COLOR_RGB2BGR))
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = _init_parser()
    parser = TrepFlow.add_model_specific_args(parser)
    args = parser.parse_args()
    model_class, ckpt_path = models_dict[args.model]
    model = model_class(args)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict)
    print(f"Using model {args.model} with checkpoint {ckpt_path}.")
    infer(args, model)
