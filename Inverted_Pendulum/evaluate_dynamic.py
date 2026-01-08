import torch
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from training_exp_mask import Dynamic
from train_dynamics import StateTransitionDataset
import math


def stat_eval(dynamic_file, data_file):

    data = torch.load(data_file)

    inputs = data["inputs"]  
    outputs = data["outputs"]

    dataset = StateTransitionDataset(inputs, outputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Dynamic(dynamic_file)

    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs_batch, targets_batch, idx in loader:
            preds = model(inputs_batch)

            # Mean squared error and mean absolute error
            mse = F.mse_loss(preds, targets_batch)#, reduction='sum')
            mae = F.l1_loss(preds, targets_batch)#, reduction='sum')

            total_mse += mse.item() * inputs_batch.size(0)
            total_mae += mae.item() * inputs_batch.size(0)
            n_samples += inputs_batch.size(0)

    avg_mse = total_mse / n_samples
    avg_mae = total_mae / n_samples

    return avg_mse, avg_mae


def make_graphs(dynamic_file, data_file, graph_out_name):

    data = torch.load(data_file)

    inputs = data["inputs"]  
    outputs = data["outputs"]

    model = Dynamic(dynamic_file)

    model.eval()

    # Get predictions
    with torch.no_grad():
        preds = model(inputs[:100])

    # Convert to numpy
    preds = preds.numpy()
    true_outputs = outputs[:100].detach().numpy()

    # Plot each output dimension
    labels = ['x\'', 'y\'', 'vx\'', 'vy\'']

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # flatten 2D array of axes into 1D list

    for i in range(4):
        ax = axes[i]
        ax.scatter(range(len(true_outputs)), true_outputs[:, i], label='True', s=25, alpha=0.6)
        ax.scatter(range(len(preds)), preds[:, i], label='Predicted', s=15, alpha=0.6)
        ax.set_title(f"{labels[i]} (Predicted vs True)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{graph_out_name}.png", dpi=300)
    print(f"Saved plot to {graph_out_name}.png")
    plt.show()

    # Compute absolute errors
    abs_errors = np.abs(preds - true_outputs)
    labels = ['x\'', 'y\'', 'vx\'', 'vy\'']

    # Plot absolute errors (no connecting lines, larger dots)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(abs_errors))

    for i in range(4):
        plt.scatter(x, abs_errors[:, i], s=30, label=f"Error in {labels[i]}", alpha=0.7)

    plt.xlabel("Sample index")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Prediction Errors per Output Dimension")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{graph_out_name}_absolute_errors_scatter.png", dpi=300)
    print(f"Saved plot to {graph_out_name}_absolute_errors_scatter.png")
    plt.show()

def get_lip_bound(dynamic_file):
        
        model = Dynamic(dynamic_file)

        model.eval()

        lip_product = 1.0
        
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight

                 # max absolute row sum = induced infinity norm
                # row_sums = torch.sum(torch.abs(W), dim=1)
                # W_inf_norm = torch.max(row_sums)

                # lip_product *= W_inf_norm.item()

                sigma = torch.linalg.norm(W, ord=2)  # Spectral norm
                lip_product *= sigma

        Linf = True

        if Linf:
            lip_product = math.sqrt(3) * lip_product 

        return lip_product

def find_max_error(dynamic_file, data_file):

    data = torch.load(data_file)

    inputs = data["inputs"]  
    outputs = data["outputs"]

    model = Dynamic(dynamic_file)

    model.eval()

    max_error = 0.0
    

    with torch.no_grad():
        for input, output in zip(inputs, outputs):
            pred = model(input)

            error = pred - output
            
            linf = torch.norm(error, p=float('inf'))

            if linf > max_error:
                max_error = linf
                max_inputs = input
                max_pred = pred
                max_outputs = output
            

    print(f"\nMax error: {max_error}\n")
    print(max_inputs)
    print(max_pred)
    print(max_outputs)

    return max_error

def find_max_error_grid(dynamic_file, spacing):
    pos_limit = 2.2
    vel_limit = 0.55
    force_limit = 1.5

    rows = []

    for x in np.arange(-pos_limit, pos_limit, spacing):
        for y in np.arange(-pos_limit, pos_limit, spacing):
            for vx in np.arange(-vel_limit, vel_limit, spacing):
                for vy in np.arange(-vel_limit, vel_limit, spacing):
                    for fx in np.arange(-force_limit, force_limit, spacing):
                        for fy in np.arange(-force_limit, force_limit, spacing):
                            point = torch.tensor([x, y, vx, vy, fx, fy], dtype=torch.float64)
                            rows.append(point)

    x_train = torch.stack(rows)

    print(x_train.shape)

    outputs_train = next_state_batch()

def find_top200_error_fast(dynamic_file, data_file, out_file, k=200):

    data = torch.load(data_file)
    inputs = data["inputs"]
    outputs = data["outputs"]


    th     = inputs[:, 0].flatten()   # shape [N]
    thdot  = inputs[:, 1].flatten()   # shape [N]
    action    = inputs[:, 2].flatten()

    u = 2 * torch.clamp(action, -1, 1)

    # feature engineering
    sin_th = torch.sin(th)
    cos_th = torch.cos(th)

    inputs = torch.stack([th, thdot, sin_th, cos_th, u], dim=1)



    model = Dynamic(dynamic_file)
    model.eval()

    all_errors = []

    max_error = 0.0

    with torch.no_grad():
        for inp, out in zip(inputs, outputs):
            pred = model(inp)
            err = torch.norm(pred - out, p=float('inf'))
            all_errors.append(err.item())

            if err > max_error:
                max_error = err


    error_tensor = torch.tensor(all_errors)

    # topk returns values and indices
    vals, idxs = torch.topk(error_tensor, k)

    top_inputs  = [inputs[i] for i in idxs]
    top_outputs = [outputs[i] for i in idxs]

    print(f"\nMax error: {max_error}\n")

    finetune_data = {
        "inputs": top_inputs,
        "outputs": top_outputs
    }

    with open(out_file, "wb") as f:
        torch.save(finetune_data, f)

    return top_inputs, top_outputs, (vals, idxs)



if __name__ == "__main__":

    # dynamic = "dynamic.pt"
    # data_eval = "val.pt"

    # pred_out_name = dynamic.split(".")[0]
    # split_out_name = data_eval.split(".")[0]
    # graph_out_name = f"{pred_out_name}_{split_out_name}"

    # mse, mae = stat_eval(dynamic, data_eval)
    # print(f"Avg MSE: {mse:.6f}")
    # print(f"Avg MAE: {mae:.6f}")

    # make_graphs(dynamic, data_eval, graph_out_name)

    # mse, mae = stat_eval("dynamics/dynamic_finetuned.pt", "dynamic_data/train_0.pt")
    # print(f"New dynamic training data MSE: {mse:.8f}")

    # find_max_error("dynamics/dynamic_finetuned.pt", "dynamic_data/train_0.pt")


    lip_bound = get_lip_bound("dynamics/test.pt")
    print("lipschitz_upper_bound: ", lip_bound )


    inps, outs, (vals, ids) = find_top200_error_fast("dynamics/test.pt", "dynamic_data/train_test.pt", "dynamic_data/finetune_test.pt", 1000)

    for inp, out, val in zip(inps, outs, vals):
        print(inp.tolist(), out.tolist(), val.item())

    import matplotlib.pyplot as plt
    import numpy as np

    inputs_t = torch.stack(inps, dim=0)      # shape: [50, 3]
    inputs_np = inputs_t.numpy() 
    errors_np = np.array(vals)

    theta = inputs_np[:, 0]
    thetadot = inputs_np[:, 1]

    theta = inputs_np[:, 0]
    thetadot = inputs_np[:, 1]

    plt.figure(figsize=(6,5))
    plt.scatter(theta, thetadot, c=errors_np, cmap='hot', s=60, edgecolor='k')
    plt.colorbar(label='Error Magnitude')
    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title("Worst 50 Errors: θ vs θ̇")
    plt.grid(True)
    plt.show()

    #find_max_error("dynamics/.pt", "dynamic_data/train_test.pt")


