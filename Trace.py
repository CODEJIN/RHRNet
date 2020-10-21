import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math

from Modules import RHRNet
from Arg_Parser import Recursive_Parse

os.environ['CUDA_VISIBLE_DEVICES']= '7'
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)


hp = Recursive_Parse(yaml.load(
    open('./Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))
model = RHRNet(hp).to(device)
print(model)

state_Dict = torch.load(
    '/data/results/RHRNet/Checkpoint/S_472000.pt',
    map_location= 'cpu'
    )
model.load_state_dict(state_Dict['Model'])
model.eval()

for param in model.parameters(): 
    param.requires_grad = False

x = torch.randn(size=(1, 2048)).to(device)
traced_model = torch.jit.trace(model, x)
traced_model.save('RHRNet.pts')