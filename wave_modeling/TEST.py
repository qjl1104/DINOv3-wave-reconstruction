import torch
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute Capability:", torch.cuda.get_device_capability(0))
