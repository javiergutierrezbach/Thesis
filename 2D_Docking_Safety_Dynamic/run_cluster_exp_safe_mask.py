from training_exp_safe_mask import train_model, retrain_model, LyapunovNetworkV, TwoDimDocking
from queries_safe_mask import safe_descent_cond_check
import torch
from generate_combined_model_torch import combined_model
from convertsinglenetwork import single_model
import os
from datetime import datetime
from parse_counterexamples import parse_counterexamples
from evaluate_dynamic import stat_eval, get_lip_bound, find_max_error
from finetune import finetune_dynamic, append_training_set, save_dynamic, save_dynamic_data
from sample_dynamics import sample_train_data, sample_grid
from train_dynamics import train_model as train_dynamic

if __name__ == "__main__":
    pos_limit = 2
    safe_pos = pos_limit - 1
    two_dim_docking = TwoDimDocking(4,4,0.5)
    vel_limit = two_dim_docking.gen_vel_limit(torch.Tensor([pos_limit]), torch.Tensor([pos_limit]))[0].numpy() + 0.01

    index = 0
    threshold = 0
    #stores dataset at out_data_file, model at out_model_file, and controller at out_controller_file
    controller_file = "controller_"
    model_file = "cert_"
    train_file = "train_"
    val_file = "val_"
    dynamic_file = f"dynamic_"
    dynamic_data_file = f"train_"
    output_comb_model = "combined_"
    initial_controller_file = "/Users/javiergutierrez/Documents/Thesis/Thesis_Repo/2D_Docking_Safety/fixed_controller_20n_manhattan.pt"

    out_controller_folders = "controllers/"
    out_model_folders = "models/"
    out_data_folders = "data/"
    out_dynamic_folders = "dynamics/"
    out_dynamic_data_folders = "dynamic_data/"
    out_comb_folders = "combined/"

    out_timing_counterexample_file = "times_counterexamples.txt"


    if not os.path.isdir(out_controller_folders):
        os.mkdir(out_controller_folders)
        os.mkdir(out_model_folders)
        os.mkdir(out_data_folders)
        os.mkdir(out_comb_folders)
        os.mkdir(out_dynamic_folders)
        os.mkdir(out_dynamic_data_folders)
        os.mkdir("counterexamples/")

    cur_train_file = out_data_folders + train_file + str(index) + ".pt"
    cur_val_file = out_data_folders + val_file + str(index) + ".pt"
    cur_model_file = out_model_folders + model_file + str(index) + ".pt"
    cur_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
    cur_controller_file = out_controller_folders + controller_file + str(index) + ".pt"
    cur_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
    cur_dynamic_file = out_dynamic_folders + dynamic_file + str(index) + ".pt"
    cur_dynamic_data_file = out_dynamic_data_folders + dynamic_data_file + str(index) + ".pt"

    dynamic_data_val_file = out_dynamic_data_folders + f"val" + ".pt"

    f = open(out_timing_counterexample_file, "w")
    print("Training initial dynamic: \n")
    f.write("Training initial dynamic: \n")
    f.close()

    st_train_time = datetime.now()

    #sample_train_data(cur_dynamic_data_file, dynamic_data_val_file)

    sample_spacing = 0.17
   # sample_grid(cur_dynamic_data_file, sample_spacing)

    #train_dynamic(cur_dynamic_data_file, cur_dynamic_file)

    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    f = open(out_timing_counterexample_file, "a")
    print("Total training time for initial dynamic", str(index), ":", str(diff.seconds))
    f.write("Total training time for initial dynamic " + str(index) + ":" + str(diff.seconds) + "\n")
    f.close()

    # Using the L-infinity norm
    dyn_lip = get_lip_bound(cur_dynamic_file)
    max_error = find_max_error(cur_dynamic_file, cur_dynamic_data_file)
    assumed_true_lip = 2.1

    error_bound = ((sample_spacing * (dyn_lip + assumed_true_lip)) / 2.0) + max_error

    f = open(out_timing_counterexample_file, "a")
    print("\nUsing L-infinity norm\n")
    f.write("\nUsing L-infinity norm\n")
    print("\nLipschitz bound for dynamic: ", str(dyn_lip), "\n")
    f.write("\nLipschitz bound for dynamic: " + str(dyn_lip) + "\n")
    print("\nMax error for dynamic: ", str(max_error), "\n")
    f.write("\nMax error for dynamic: " + str(max_error) + "\n")
    print("\nAssumed lip for true dynamic: ", str(assumed_true_lip), "\n")
    f.write("\nAssumed lip for true dynamic: " + str(assumed_true_lip) + "\n")
    print("\nSpacing for grid sampling: ", str(sample_spacing), "\n")
    f.write("\nSpacing for grid sampling: " + str(sample_spacing) + "\n")
    print("\nError on dynamic bounded by: ", str(error_bound), "\n")
    f.write("\nError on dynamic bounded by: " + str(error_bound)+ "\n")
    f.close()

    threshold = 0
    st_train_time = datetime.now()
    train_model(safe_pos, pos_limit, vel_limit, cur_train_file, cur_val_file, cur_model_file, cur_controller_file, threshold, initial_controller_file, cur_dynamic_file)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    f = open(out_timing_counterexample_file, "a")
    print("Total training time for model index", str(index), ":", str(diff.seconds))
    f.write("Total training time for model index " + str(index) + ":" + str(diff.seconds) + "\n")
    f.close()
    combined_model(cur_model_file, cur_controller_file, cur_dynamic_file, cur_comb_file)
    single_model(cur_model_file, cur_model_onnx_file)
    st_ver_time = datetime.now()
    ret, ret_ranges, failed = safe_descent_cond_check(cur_comb_file, cur_model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = vel_limit)
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    f = open(out_timing_counterexample_file, "a")
    print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds))
    f.write("Total verification time for verification index " + str(index) + ":" + str(diff_ver_time.seconds) + "\n")
    f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
    f.write("Number of failed cases: " + str(len(failed)) + "\n")
    f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
    f.close()
    while (len(ret) > 0):
        index += 1
        next_train_file = out_data_folders + train_file + str(index) + ".pt"
        next_val_file = out_data_folders + val_file + str(index) + ".pt"
        next_model_file = out_model_folders + model_file + str(index) + ".pt"
        next_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
        next_controller_file = out_controller_folders + controller_file + str(index) + ".pt"
        next_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
        next_dynamic_file = out_dynamic_folders + dynamic_file + str(index) + ".pt"
        next_dynamic_data_file = out_dynamic_data_folders + dynamic_data_file + str(index) + ".pt"

        save_dynamic(cur_dynamic_file, next_dynamic_file)
        save_dynamic_data(cur_dynamic_data_file, next_dynamic_data_file)

        st_train_time = datetime.now()
        retrain_model(index - 1, torch.Tensor(ret), torch.Tensor(ret_ranges), safe_pos, pos_limit, vel_limit, cur_train_file, next_train_file, next_val_file, cur_model_file, cur_controller_file, next_model_file, next_controller_file, next_dynamic_file, threshold)
        end_train_time = datetime.now()
        diff = end_train_time - st_train_time
        f = open(out_timing_counterexample_file, "a")
        print("Total training time for model index", str(index), ":", str(diff.seconds))
        f.write("Total training time for model index " + str(index) + ":" + str(diff.seconds) + "\n")
        f.close()
        combined_model(next_model_file, next_controller_file, cur_dynamic_file, next_comb_file)
        single_model(next_model_file, next_model_onnx_file)
        st_ver_time = datetime.now()
        ret, ret_ranges, failed = safe_descent_cond_check(next_comb_file, next_model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = vel_limit)
        end_ver_time = datetime.now()
        diff_ver_time = end_ver_time - st_ver_time
        f = open(out_timing_counterexample_file, "a")
        print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
        f.write("Total training time for verification index " + str(index) + ":" + str(diff_ver_time.seconds) + "\n")
        f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
        f.write("Number of failed cases: " + str(len(failed)) + "\n")
        f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
        f.close()
        cur_train_file, cur_val_file, cur_model_file, cur_controller_file, cur_dynamic_file, cur_dynamic_data_file = next_train_file, next_val_file, next_model_file, next_controller_file, next_dynamic_file, next_dynamic_data_file

    print(failed)