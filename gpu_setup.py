# Setup and settings for GPU training
import torch

if not torch.cuda.is_available():
    print("No GPU available")
else:
    setup_GPU()

def setup_GPU(self):
    this.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(this.device)

