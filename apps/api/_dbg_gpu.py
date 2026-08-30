import sys
import time

import torch

from training.pipeline import IMG_SIZE, _build_unet

amp = sys.argv[1] == "amp"
bench = len(sys.argv) > 2 and sys.argv[2] == "bench"
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.backends.cudnn.benchmark = bench

net = _build_unet().cuda()
print("params(M)=%.2f" % (sum(p.numel() for p in net.parameters()) / 1e6))
if amp:
    net = net.to(memory_format=torch.channels_last)
opt = torch.optim.Adam(net.parameters(), 1e-3)
bce = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=amp)
x = torch.rand(batch, 3, IMG_SIZE, IMG_SIZE, device="cuda")
if amp:
    x = x.contiguous(memory_format=torch.channels_last)
y = (torch.rand(batch, 1, IMG_SIZE, IMG_SIZE, device="cuda") > 0.9).float()

for i in range(8):
    if i == 3:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        loss = bce(net(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"amp={amp} bench={bench} batch={batch}: {5 * batch / dt:6.1f} img/s  "
      f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
