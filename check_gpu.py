import platform
import torch

print("=== Environment ===")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("\nRESULT: GPU unavailable in this environment.")
    print("CPU development and testing can continue.")
    raise SystemExit(0)

print("\n=== GPU Details ===")
print("GPU count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"\nGPU {index}: {props.name}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1024**3:.2f} GiB")

print("\nRESULT: CUDA is available.")
