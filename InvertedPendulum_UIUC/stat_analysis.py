import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from training_exp_mask import Controller, LyapunovNetworkV


def sample_data(n):
    x = None
    while x is None or len(x) < n:
        x_temp = torch.Tensor(
            n, 2).uniform_(-0.3, 0.3)
        safe_mask = abs(x_temp[:, 0]) >= 0.21
        safe_mask.logical_or_(abs(x_temp[:, 1]) >= 0.21)
        if x is None:
            x = x_temp[safe_mask]
        else:
            x = torch.cat((x, x_temp[safe_mask]))

    x = x[:n]
    return x.clone().detach()


def attack_pgd(x0, V, D=0.005, alpha=0.005, steps=10):
    """Perform Projected Gradient Descent (PGD) to maximize V(x') around x."""
    # Clone the input tensor and prepare for gradient computation
    x_adv = x0.clone().detach().requires_grad_(True)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = V(x_adv)
        loss.backward(torch.ones(x_adv.shape[0]).reshape(-1, 1))
        with torch.no_grad():
            # Step in the direction that maximizes V (opposite of the loss's gradient)
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # Project the perturbation to the epsilon ball around x (L-infinity norm)
            perturbation = x_adv - x0
            perturbation = torch.clamp(perturbation, min=-D, max=D)
            x_adv = x0 + perturbation

            # Ensure the adversarial example stays within valid input range
            x_adv = torch.clamp(x_adv, min=-0.7, max=0.7)

    # Detach the final adversarial example to prevent gradient leakage
    return x_adv.detach()


def simulate_steps(x_init, controller, V):
    """Simulate multiple steps until all points reach the goal region."""
    num_samples = x_init.shape[0]
    steps = torch.zeros(num_samples, dtype=torch.int32)
    # Track which points are still moving
    active = torch.ones(num_samples, dtype=torch.bool)
    x = x_init.clone()
    avg_step_length = torch.zeros(num_samples, dtype=torch.float32)
    Failed = False
    V_step = []
    # all_x = [x.clone()]
    all_x = []
    violate = 0
    while active.any() and max(steps) < 200:
        if max(steps) % 10 == 0:
            print(max(steps))
            # print(x[0:10])
        # x_adv = attack_rand(x, V)
        # x_adv = attack_pgd_x0(x, controller, V)
        # actions = controller.nn(x_adv)
        # x = attack_pgd_x0(x, controller, V)
        # Forward step using controller
        x_next = controller.next_step(x)
        # print(controller.nn(x_next)[:5])
        # x_next = attack_pgd(x_next, V)
        # if active[0]:
        #     print("x", x[0], V(x[0]), "x_next", x_next[0], V(x_next[0]))
        V_step.append((V(x_next)-V(x)).sum())
        # print("lip", local_lipschitz_calculate(V, x_next))
        step_length = torch.max(
            torch.abs((torch.abs(x_next) - torch.abs(x))), dim=1)[0]
        # print(min(step_length[active]))
        # print(x_next[0:10])
        x_next = attack_pgd(x0=x_next, V=V)
        # print("after", x_next[0:5])
        # x_next = attack_square(x0=x_next, model=controller)

        # x_next = attack_rand(x=x_next, V=V)
        # L-infinity step length
        # step_length = torch.max(torch.abs((x_next - x)), dim=1)[0]
        # print(torch.amax(x-x_next, dim=1).shape)
        # print(torch.zeros(x.shape[0]))
        # print("x-x_n", torch.amax(torch.abs(x)-torch.abs(x_next), dim=1))
        # violate += -torch.minimum(torch.zeros(x[active].shape[0]),
        #                           torch.amax(
        #     (torch.abs(x[active]) - torch.abs(x_next[active])), dim=1)).sum().item()
        # print(violate)
        avg_step_length[active] = (
            step_length[active] + avg_step_length[active]*steps[active])/(steps[active]+1)
        new_reached_goal = (x_next[:, 0].abs() <= 0.2 + 0.011) & (
            x_next[:, 1].abs() <= 0.2 + 0.011)
        steps[active & new_reached_goal] += 1
        active &= ~new_reached_goal  # Mark those that reached goal as inactive
        x = x_next  # Update positions
        all_x.append(x_next[active].clone())
        steps[active] += 1  # Increment step count for active samples
        # print(V(x_next)[:5], V(x)[:5])
    # print(V_step)
    if active.any():
        print(
            f"{sum(active)} samples not reached goal, {sum(~active)} reached goal")
        return sum(~active) / num_samples
        Failed = True
    else:
        print("All samples reached goal")
        return 1.0


groups = {
    "vanilla": "controllers_epsl_5e-3{}",
    "lip_L": "controllers_epsl_5e-3_lip3{}",
    "lipneighbor": "controllers_epsl_5e-3_lipneighbor1e-3{}",
    "pgd": "controllers_epsl_5e-3_pgd5e-3{}"
}

group_suffixes = {
    "vanilla": ['', '_2', '_3', '_4', '_5', '_6', '_7'],
    "lip_L": ['', '_2', '_3', '_4', '_5', '_6', '_7'],
    # "lip_L": ['', '_2', '_3', '_5', '_7'],
    # "lipneighbor": ['1', '2', '3', '4', '5'],
    "lipneighbor": ['', '_2', '_3', '_4', '_6', '_7'],
    "pgd": ['', '_2', '_3', '_4', '_5']
}
x_init = sample_data(1000)
results = {}

for group_name, template in groups.items():
    suffixes = group_suffixes[group_name]
    rates = []

    for suffix in suffixes:
        ctrl_folder = template.format(suffix)
        if not os.path.isdir(ctrl_folder):
            print(f"Missing folder: {ctrl_folder}")
            continue

        # Find the controller_*.pt with the largest number
        ctrl_files = [f for f in os.listdir(
            ctrl_folder) if re.match(r'controller_(\d+)\.pt', f)]
        if not ctrl_files:
            continue

        ctrl_nums = [int(re.search(r'(\d+)', f).group()) for f in ctrl_files]
        max_num = max(ctrl_nums)
        ctrl_path = os.path.join(ctrl_folder, f'controller_{max_num}.pt')

        # Now find corresponding model folder
        model_folder = ctrl_folder.replace("controllers", "models")
        if not os.path.isdir(model_folder):
            print(f"Missing model folder: {model_folder}")
            continue

        cert_path = os.path.join(model_folder, f'cert_{max_num}.pt')
        if not os.path.isfile(cert_path):
            print(f"Missing cert file: {cert_path}")
            continue

        # Load files
        controller = Controller(
            ctrl_path, isInitial=False, t=1, device="cpu")
        print(f"Loaded controller from {ctrl_path}")
        V = torch.load(cert_path, weights_only=False, map_location="cpu")

        # Run simulation
        rate = simulate_steps(x_init, controller, V)
        rates.append(rate)

    if rates:
        results[group_name] = rates
    else:
        print(f"No valid runs for {group_name}")

print(results)

# Plotting
# group_names = list(results.keys())
# means = [np.mean(results[g]) for g in group_names]
# stds = [np.std(results[g]) for g in group_names]

# x = np.arange(len(group_names))
# plt.figure(figsize=(8, 6))
# bars = plt.bar(x, means, yerr=stds, capsize=5)
# plt.xticks(x, group_names)
# plt.ylabel("Rate")
# plt.title("Simulation Results by Group")

# # Annotate bars
# for i, (mean, std) in enumerate(zip(means, stds)):
#     plt.text(i, mean + 0.001, f'{mean:.4f}', ha='center')

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("simulation_results.png")
