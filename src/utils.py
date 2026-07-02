import random
import warnings

import numpy as np
import torch
import torchvision

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory     : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"CUDA Version   : {torch.version.cuda}")
    else:
        print("No GPU detected. Training will run on CPU.")

    return device


def print_environment():
    print(f"PyTorch      : {torch.__version__}")
    print(f"Torchvision  : {torchvision.__version__}")