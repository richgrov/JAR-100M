import torch

if torch.cuda.is_available():
    print("Running on cuda")
    device = torch.device("cuda")
else:
    print("Running on cpu")
    device = torch.device("cpu")

