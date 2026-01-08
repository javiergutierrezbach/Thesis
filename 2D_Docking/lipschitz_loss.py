import torch
import torch.nn.functional as F

def lipschitz_loss(cert_file, lipschitz_factor):
        
        V = torch.load(cert_file)
        
        lip_product = 1.0
        
        for layer in V.modules():
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight
                sigma = torch.linalg.norm(W, ord=2)  # Spectral norm
                lip_product *= sigma

        lipschitz_violation = F.relu(lip_product - 1)
        lipschitz_term = lipschitz_factor * lipschitz_violation

        return lipschitz_term

if __name__ == "__main__":
     
     cert_file = "models/cert_lip_4.pt"

     loss = lipschitz_loss(cert_file, 1)

     print(f"{loss.item()}")
     