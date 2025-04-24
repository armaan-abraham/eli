import torch


torch.cuda.memory._record_memory_history(
    max_entries=100
)

x = torch.randn(1000, 1000, device="cuda")

del x

snapshot = torch.cuda.memory._snapshot()

print("done")

