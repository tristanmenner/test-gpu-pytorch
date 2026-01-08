import torch
import numpy as np

def main() -> int:
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {device}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is NOT available. Falling back to CPU.")

    # Create a tensor and move it to the selected device
    x = torch.randn(3, 3)
    x_device = x.to(device)

    print(f"Tensor on {device}:\n{x_device}")

    # Move back to CPU and do a simple operation, just to prove it works
    x_cpu = x_device.to("cpu")
    print(f"Tensor moved back to CPU, mean value: {x_cpu.mean().item():.6f}")

    return 0
    




if __name__ == "__main__":
    raise SystemExit(main())