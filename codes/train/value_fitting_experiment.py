import torch
from model_ext.baseline_models import FeatureExtractor_baseline
from types import SimpleNamespace


import yaml
with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
flags = config
loc='/home/pooja/PycharmProjects/lux_ai/outputs/inputs/'
states=torch.load(loc+'state_array.pt')
values=torch.load(loc+'vals_array.pt')
model=FeatureExtractor_baseline(config.model_exp)