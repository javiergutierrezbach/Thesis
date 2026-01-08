import torch

controller_path = "controllers/controller_lip_5_verified.pt"

controller = torch.load(controller_path)

inputs = torch.tensor([-1.1029335754227094, -2.1999, -0.18117530347314043, -0.17266216863730233], dtype=torch.float64)

outputs = controller(inputs)

print(outputs)