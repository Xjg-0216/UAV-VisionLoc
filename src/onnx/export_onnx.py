

from model.network import GeoLocalizationNet
import cv2
import sys
import parser
import numpy as np
import torch
import torch.onnx
from torch import nn
import argparse
import onnxruntime
from onnxruntime.datasets import get_example


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



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
    save_output =  "/media/bh/xujg/UAV-VisionLoc/src/uvl2.onnx" 
    print(model)
    x = torch.randn(1, 3, 512, 512, device='cpu')
    torch.onnx.export(model, x, save_output, export_params=19)


    # demo data
    dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
    example_model = get_example(save_output)
    # 指定执行提供程序
    providers = ['CPUExecutionProvider']  # 也可以根据需要添加 'AzureExecutionProvider'
    
    # 创建推理会话
    session = onnxruntime.InferenceSession(example_model, providers=providers)

    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: to_numpy(dummy_input)})

    #onnx output
    print("onnx_output: ", onnx_output)
    print("onnx_output shape: ", onnx_output[0].shape)

    #pytorch model network output
    with torch.no_grad():
        torch_out = model(dummy_input)
    print("pytorch_out: ", torch_out)
    print("pytorch_output shape: ", torch_out.shape)
