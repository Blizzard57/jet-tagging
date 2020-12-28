import torch

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
PROJECT_DIR = "scratch/"
