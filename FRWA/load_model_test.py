import torch

try:
    data = torch.load("fixed_controller_20n_manhattan.pt")
    print("Loaded successfully:", type(data))
    print(data)
except Exception as e:
    print("Failed to load:", e)
