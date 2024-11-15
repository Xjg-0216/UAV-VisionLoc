

from model.network import GeoLocalizationNet
import cv2
import numpy as np
import torch
import torch.onnx
from torch import nn
import argparse
import onnxruntime
import parser
from onnxruntime.datasets import get_example


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def parse_opt(known=False):
    """Parses command-line arguments for VTL model export configurations, returning the parsed options."""
    parser = argparse.ArgumentParser()
    #### model
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18conv4",
        choices=[
            "alexnet",
            "vgg16",
            "resnet18conv4",
            "resnet18conv5",
            "resnet50conv4",
            "resnet50conv5",
            "resnet101conv4",
            "resnet101conv5",
            "cct384",
            "vit",
        ],
        help="_")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="netvlad",
        choices=[
            "netvlad",
            "gem",
            "spoc",
            "mac",
            "rmac",
            "crn",
            "rrm",
            "cls",
            "seqpool",
            "none",
        ],
    )
    parser.add_argument(
        "--netvlad_clusters",
        type=int,
        default=64,
        help="Number of clusters for NetVLAD layer.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="PCA dimension (number of principal components). If None, PCA is not used.",
    )
    parser.add_argument("--non_local", action="store_true", help="_")
    parser.add_argument(
        "--channel_bottleneck",
        type=int,
        default=128,
        help="Channel bottleneck for Non-Local blocks",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=None,
        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.",
    )
    parser.add_argument(
        "--conv_output_dim",
        type=int,
        default=4096,
        help="Output dimension of conv layer. If None, don't use a conv layer.",
    )
    parser.add_argument(
        "--unfreeze",
        action='store_true',
        help="Unfreeze the first few layers for backbone",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="imagenet",
        choices=["imagenet", "gldv2", "places", "none"],
        help="Select the pretrained weights for the starting network",
    )
    parser.add_argument(
        "--DA",
        type=str,
        default='none',
        choices=['none', 'DANN_before', 'DANN_after', 'DANN_before_conv'],
        help="Domain adaptation"
    )
    parser.add_argument(
        "--add_bn",
        action="store_true",
        default=True,
        help="Add bn to compression layers"
    )
    parser.add_argument(
        "--remove_relu",
        action="store_true",
        help="Remove last relu layer of backbone"
    )
    parser.add_argument(
        "--l2",
        type=str,
        default="before_pool",
        choices=["before_pool", "after_pool", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    parser.add_argument(
        "--trunc_te", type=int, default=None, choices=list(range(0, 14))
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=[512, 512],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )


    parser.add_argument("--weights", nargs="+", type=str, default="/home/xujg/code/satellite-thermal-geo-localization/last_model.pth", help="model.pth path(s)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt




def init_torch_model(args):
    model = GeoLocalizationNet(args)
    state_dict = torch.load(args.resume, map_location='cpu')['model_state_dict']

    #Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(state_dict)
    model.eval()
    return model



if __name__ == "__main__":
    args = parser.parse_arguments()
    model = init_torch_model(args)
    print(model)
    x = torch.randn(1, 3, 512, 512, device='cpu')
    save_output = "/root/temp/uvl_926.onnx"
    torch.onnx.export(model, x, save_output, export_params=19)


    # demo data
    dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
    example_model = get_example(save_output)

    session = onnxruntime.InferenceSession(example_model)

    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: to_numpy(dummy_input)})

    #onnx output
    print("onnx_output: ", onnx_output)
    print("onnx_output shape: ", onnx_output[0].shape)

    #pytorch model network output

    torch_out = model(dummy_input)
    print("pytorch_out: ", torch_out)
    print("pytorch_output shape: ", torch_out.shape)
















