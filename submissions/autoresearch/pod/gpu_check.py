"""Is the GPU healthy? A cheap host can show a GPU in nvidia-smi that PyTorch cannot use.

This trains one small step on the GPU and on the CPU from the same weights and checks
that both give the same numbers (float32, TF32 off), then times a convolution.
"""

import sys
import time

import torch
from torch import nn

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch sees no GPU"
print("gpu", torch.cuda.get_device_name())
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(0)
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10),
)
x, y = torch.rand(64, 3, 64, 64), torch.randint(0, 10, (64,))
grads = {}
for device in ("cpu", "cuda"):
    model = net.to(device)
    model.zero_grad()
    nn.functional.cross_entropy(model(x.to(device)), y.to(device)).backward()
    grads[device] = model[0].weight.grad.detach().cpu().clone()
diff = (grads["cpu"] - grads["cuda"]).abs().max().item()
print(f"largest GPU-CPU gradient difference {diff:.2e}")
assert diff < 1e-4, "the GPU computes different numbers from the CPU"

conv = nn.Conv2d(64, 64, 3, padding=1).cuda()
batch = torch.rand(256, 64, 32, 32, device="cuda")
for _ in range(3):
    conv(batch)
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    conv(batch)
torch.cuda.synchronize()
flops = 20 * 2 * 256 * 32 * 32 * 64 * 64 * 9
print(f"conv speed {flops / (time.time() - start) / 1e12:.1f} TFLOP/s (float32)")
print("GPU OK")
