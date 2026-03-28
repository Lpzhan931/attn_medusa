import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    BASE_MODEL_PATH = "/home/share/models/Qwen3-8B/"
