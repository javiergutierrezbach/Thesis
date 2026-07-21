import pickle
from training_exp_mask import train_model, retrain_model, LyapunovNetworkV, InvertedPendulum
from queries_mask import safe_descent_cond_check
import torch
from generate_combined_model_torch import combined_model
from convertsinglenetwork import single_model
import os
from datetime import datetime



def main():
    # also change the pltrainer in train_model
    #device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    is_traj = False
    pos_limit = 0.7
    safe_pos = 0.3
    index = 0
    threshold = 0
    # stores dataset at out_data_file, model at out_model_file, and controller at out_controller_file
    controller_file = "controller_"
    model_file = "cert_"
    train_file = "train_"
    val_file = "val_"
    output_comb_model = "combined_"
    initial_controller_file = "pend_100_ppo_128.pth"

    out_controller_folders = "controllers_epsl_5e-3_lip3_4/"
    out_model_folders = "models_epsl_5e-3_lip3_4/"
    out_data_folders = "data_epsl_5e-3_lip3_4/"
    out_comb_folders = "combined_epsl_5e-3_lip3_4/"
    counterexample_folder = "counterexamples_epsl_5e-3_lip3_4/"

    out_timing_counterexample_file = counterexample_folder + "time.txt"
    if not os.path.isdir(out_controller_folders):
        os.mkdir(out_controller_folders)
        os.mkdir(out_model_folders)
        os.mkdir(out_data_folders)
        os.mkdir(out_comb_folders)
        os.mkdir(counterexample_folder)

    cur_train_file = out_data_folders + train_file + str(index) + ".pt"
    cur_val_file = out_data_folders + val_file + str(index) + ".pt"
    cur_model_file = out_model_folders + model_file + str(index) + ".pt"
    cur_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
    cur_controller_file = out_controller_folders + \
        controller_file + str(index) + ".pt"
    cur_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
    st_train_time = datetime.now()
    train_model(cur_train_file, cur_val_file,
                cur_model_file, cur_controller_file, threshold, initial_controller_file, out_timing_counterexample_file, device)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    f = open(out_timing_counterexample_file, "a")
    print("Total training time for model index",
        str(index), ":", str(diff.seconds))
    f.write("Total training time for model index " +
            str(index) + ":" + str(diff.seconds) + "\n")
    f.close()

    combined_model(cur_model_file, cur_controller_file, cur_comb_file, device)
    single_model(cur_model_file, cur_model_onnx_file, device)
    st_ver_time = datetime.now()
    ret, ret_ranges, failed = safe_descent_cond_check(
        cur_comb_file, cur_model_onnx_file, safe_pos=safe_pos, limit_pos=pos_limit, docking_pos=0.2)
    print(len(ret))
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    f = open(out_timing_counterexample_file, "a")
    print("Total verification time for verification index",
        str(index), ":", str(diff_ver_time.seconds))
    f.write("Total verification time for verification index " +
            str(index) + ":" + str(diff_ver_time.seconds) + "\n")
    f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
    f.write("Number of failed cases: " + str(len(failed)) + "\n")
    f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
    f.close()

    variables = {
        "pos_limit": pos_limit,
        "safe_pos": safe_pos,
        "index": index,
        "threshold": threshold,
        "controller_file": controller_file,
        "model_file": model_file,
        "train_file": train_file,
        "val_file": val_file,
        "output_comb_model": output_comb_model,
        "initial_controller_file": initial_controller_file,
        "out_controller_folders": out_controller_folders,
        "out_model_folders": out_model_folders,
        "out_data_folders": out_data_folders,
        "out_comb_folders": out_comb_folders,
        "counterexample_folder": counterexample_folder,
        "out_timing_counterexample_file": out_timing_counterexample_file,
        "cur_train_file": cur_train_file,
        "cur_val_file": cur_val_file,
        "cur_model_file": cur_model_file,
        "cur_model_onnx_file": cur_model_onnx_file,
        "cur_controller_file": cur_controller_file,
        "cur_comb_file": cur_comb_file,
        "ret": ret,
        "ret_ranges": ret_ranges,
        "failed": failed
    }
    with open("checkpoint.pkl", "wb") as f:
        pickle.dump(variables, f)

    print("Checkpoint saved successfully!")
    # exit()
    with open("checkpoint.pkl", "rb") as f:
        variables = pickle.load(f)
    pos_limit = variables["pos_limit"]
    safe_pos = variables["safe_pos"]
    index = variables["index"]
    threshold = variables["threshold"]
    controller_file = variables["controller_file"]
    model_file = variables["model_file"]
    train_file = variables["train_file"]
    val_file = variables["val_file"]
    output_comb_model = variables["output_comb_model"]
    initial_controller_file = variables["initial_controller_file"]
    out_controller_folders = variables["out_controller_folders"]
    out_model_folders = variables["out_model_folders"]
    out_data_folders = variables["out_data_folders"]
    out_comb_folders = variables["out_comb_folders"]
    out_timing_counterexample_file = variables["out_timing_counterexample_file"]
    cur_train_file = variables["cur_train_file"]
    cur_val_file = variables["cur_val_file"]
    cur_model_file = variables["cur_model_file"]
    cur_model_onnx_file = variables["cur_model_onnx_file"]
    cur_controller_file = variables["cur_controller_file"]
    cur_comb_file = variables["cur_comb_file"]
    ret = variables["ret"]
    ret_ranges = variables["ret_ranges"]
    failed = variables["failed"]
    counterexample_folder = variables["counterexample_folder"]
    traj_file = "traj_"
    cur_traj_file = out_data_folders + traj_file + str(index) + ".pt"

    # counterexample_folder = "ptb_li_d_0015_counterexamples/"
    while (len(ret) > 0 or is_traj == True):
        # with open('log_d0001.txt', 'a') as f:
        #     f.write("Model index: " + str(index) + "\n")
        #     f.write("Number of counterexamples: " + str(len(ret)) + "\n")
        index += 1
        # next_train_file = out_data_folders + train_file + str(index) + ".pt"
        next_val_file = out_data_folders + val_file + str(index) + ".pt"
        next_train_file = cur_train_file
        next_val_file = cur_val_file
        next_traj_file = out_data_folders + traj_file + str(index) + ".pt"
        next_model_file = out_model_folders + model_file + str(index) + ".pt"
        next_model_onnx_file = out_model_folders + \
            model_file + str(index) + ".onnx"
        next_controller_file = out_controller_folders + \
            controller_file + str(index) + ".pt"
        next_comb_file = out_comb_folders + \
            output_comb_model + str(index) + ".onnx"
        st_train_time = datetime.now()
        retrain_model(index - 1, torch.Tensor(ret), torch.Tensor(ret_ranges), cur_train_file, cur_traj_file,
                    next_train_file, next_val_file, next_traj_file, cur_model_file, cur_controller_file, next_model_file, next_controller_file, counterexample_folder, out_timing_counterexample_file, threshold, device=device)
        end_train_time = datetime.now()
        diff = end_train_time - st_train_time
        f = open(out_timing_counterexample_file, "a")
        print("Total training time for model index",
            str(index), ":", str(diff.seconds))
        f.write("Total training time for model index " +
                str(index) + ":" + str(diff.seconds) + "\n")
        f.close()
        combined_model(next_model_file, next_controller_file,
                    next_comb_file, device=device)
        single_model(next_model_file, next_model_onnx_file, device=device)
        st_ver_time = datetime.now()
        ret, ret_ranges, failed = safe_descent_cond_check(
            next_comb_file, next_model_onnx_file, safe_pos=safe_pos, limit_pos=pos_limit, docking_pos=0.2)

        end_ver_time = datetime.now()
        diff_ver_time = end_ver_time - st_ver_time
        f = open(out_timing_counterexample_file, "a")
        print("Total verification time for verification index",
            str(index), ":", str(diff_ver_time.seconds) + "\n")
        f.write("Total training time for verification index " +
                str(index) + ":" + str(diff_ver_time.seconds) + "\n")
        f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
        f.write("Number of failed cases: " + str(len(failed)) + "\n")
        f.write("Verification counterexamples: " + str(ret) + "\n")
        f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
        f.close()
        cur_train_file, cur_val_file, cur_model_file, cur_controller_file = next_train_file, next_val_file, next_model_file, next_controller_file
        variables = {
            "pos_limit": pos_limit,
            "safe_pos": safe_pos,
            "index": index,
            "threshold": threshold,
            "controller_file": controller_file,
            "model_file": model_file,
            "train_file": train_file,
            "val_file": val_file,
            "output_comb_model": output_comb_model,
            "initial_controller_file": initial_controller_file,
            "out_controller_folders": out_controller_folders,
            "out_model_folders": out_model_folders,
            "out_data_folders": out_data_folders,
            "out_comb_folders": out_comb_folders,
            "counterexample_folder": counterexample_folder,
            "out_timing_counterexample_file": out_timing_counterexample_file,
            "cur_train_file": cur_train_file,
            "cur_val_file": cur_val_file,
            "cur_model_file": cur_model_file,
            "cur_model_onnx_file": cur_model_onnx_file,
            "cur_controller_file": cur_controller_file,
            "cur_comb_file": cur_comb_file,
            "ret": ret,
            "ret_ranges": ret_ranges,
            "failed": failed
        }
        with open("checkpoint1.pkl", "wb") as f:
            pickle.dump(variables, f)

        print("Checkpoint saved successfully!")
    print(failed)

if __name__ == "__main__":
    main()