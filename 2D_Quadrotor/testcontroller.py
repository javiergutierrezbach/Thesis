from attempt_conversion import LearnedController
import torch
from torch import nn
from training_exp_mask import Dynamic
from sample_dynamics import quad_next_state_batch

file_name = "init_controller.pth"

net = LearnedController().double().to()
net.load_state_dict(torch.load(file_name, weights_only=False, map_location=torch.device("cpu")))


#file_name = "controller_initial.pt"
# net = nn.Sequential(
#             nn.Linear(6, 6, bias=True),
#             nn.LeakyReLU(0.01),
#             nn.Linear(6, 4, bias=True),
#             nn.LeakyReLU(0.01),
#             nn.Linear(4, 2, bias=True),
#         ).double()

# ckpt = torch.load(file_name, map_location=torch.device("cpu"))
# net.load_state_dict(ckpt["state_dict"])

dt = 0.01
length=0.25 
mass=0.486 
inertia=0.00383 
gravity=9.81

u_hover = mass * gravity / 2.0  # scalar
hover_vec = torch.tensor([[u_hover, u_hover]], dtype=torch.float64)

dynamic = Dynamic("dynamics/dynamic_0.pt")

# net must be loaded already (your controller)
net.eval()

def run_case(name, x):
    x = torch.tensor([x], dtype=torch.float64)  # (1,6)
    with torch.no_grad():
        a = net(x)  # raw controller output
        u = torch.clamp(a + hover_vec, 0.0, 6.0)  # your mapping

    u1, u2 = u[0,0].item(), u[0,1].item()
    print(f"\n{name}")
    print("x =", x.numpy().squeeze().tolist())
    print("raw a =", a.numpy().squeeze().tolist())
    print("u =", [u1, u2], " sum =", u1+u2, " diff =", u1-u2, " mg =", mass*gravity)

    inp = torch.cat([x, u], dim=1)      # (1,8)

    x1 = dynamic(inp)                     # predicted next state
    x1_real = quad_next_state_batch(inp)
    print("x1_dynamic:", x1)
    print("x1_real:", x1_real)

# 1) perfect hover
run_case("1) hover rest", [0,0,0, 0,0,0])

# 2) falling down: expect more thrust sum than hover
run_case("2) falling (vz = -1.0)", [0,0,0, 0,-1.0,0])

# 3) rising: expect less thrust sum
run_case("3) rising (vz = +1.0)", [0,0,0, 0,+1.0,0])

# 4) tilted right: expect corrective torque (diff nonzero) + often a bit more thrust
run_case("4) theta = +0.2 rad", [0,0,0.2, 0,0,0])

# 5) spinning positive: expect opposite torque (diff that damps omega)
run_case("5) omega = +0.5", [0,0,0, 0,0, +0.5])

# 6) moving right: expect it to eventually brake (may command torque)
run_case("6) vx = +1.0", [0,0,0, +1.0,0,0])