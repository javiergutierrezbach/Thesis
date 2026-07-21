from attempt_conversion import LearnedController
import torch
from torch import nn
from training_exp_mask import Dynamic


cert_name = "models/cert__lip_0_e-2robust.pt"
controller_name = "controllers/controller__lip_0_e-4robust.pt"

cert = torch.load(cert_name, weights_only=False, map_location=torch.device("cpu"))

controller = torch.load(controller_name, weights_only=False, map_location=torch.device("cpu"))

dynamic = Dynamic("dynamics/dynamic__lip_0.pt")


# net must be loaded already (your controller)
cert.eval()
controller.eval()
dynamic.eval()

def run_case(name, x):
    x = torch.tensor([x], dtype=torch.float64)  # (1,4)
    with torch.no_grad():
        a = cert(x)  # raw controller output
        u = controller(x)  # raw controller output
        u = torch.clamp(u, -1.0, 1.0)  # your mapping
        inp = torch.cat([x, u], dim=1)
        x1 = dynamic(inp)
        u = controller(x1)
        u = torch.clamp(u, -1.0, 1.0)  # your mapping
        inp1 = torch.cat([x1, u], dim=1)
        x2 = dynamic(inp1)

        a1 = cert(x1)
        a2 = cert(x2)


    print(f"\n{name}")
    print("x =", x.numpy().squeeze().tolist())
    print("cert value =", a.numpy().squeeze().tolist())
    print("input =", inp.numpy().squeeze().tolist())
    print("x1 =", x1.numpy().squeeze().tolist())
    print("cert1 value =", a1.numpy().squeeze().tolist())
    print("input1 =", inp1.numpy().squeeze().tolist())
    print("x2 =", x2.numpy().squeeze().tolist())
    print("cert2 value =", a2.numpy().squeeze().tolist())
    print("\n")



# 1) perfect hover    
run_case("1) center", [0,0,0,0])

# 2) falling down: expect more thrust sum than hover
run_case("2) moving left (vx = -1.0)", [0.3, 0.3,0.4,0])

# 3) rising: expect less thrust sum
run_case("3) moving up (vy = +1.0)", [0,0,0,0.4])

# 4) tilted right: expect corrective torque (diff nonzero) + often a bit more thrust
run_case("4) safe zone", [1,1,0.3,0])

# 5) spinning positive: expect opposite torque (diff that damps omega)
run_case("5) unsafe ", [2.2,1.9,0.7,-0.3])

# 6) moving right: expect it to eventually brake (may command torque)
run_case("6) unsafe all", [5,-5,2, 1.0])

# 6) moving right: expect it to eventually brake (may command torque)
run_case("7) near unsafe ", [1.9, 1.8, 0.3, 0.4])

# 6) moving right: expect it to eventually brake (may command torque)
run_case("8) counterexample ", [-0.35, -1.18, -0.25, 0.0])

