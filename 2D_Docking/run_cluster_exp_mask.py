from training_exp_mask import train_model, retrain_model, TwoDimDocking
from queries_mask import safe_descent_cond_check
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


    lip_loss = True

    if lip_loss:
        lip_ext = "_lip_"
    else:
        lip_ext = ""

    index = 0
    threshold = 0
    #stores dataset at out_data_file, model at out_model_file, and controller at out_controller_file
    controller_file = f"controller_{lip_ext}"
    model_file = f"cert_{lip_ext}"
    controller_data_file = f"train_{lip_ext}"
    val_file = f"val_{lip_ext}"
    output_comb_model = f"combined_{lip_ext}"
    dynamic_file = f"dynamic_{lip_ext}"
    counterexample_file = f"counterexample_{lip_ext}"
    dynamic_data_file = f"train_{lip_ext}"
    initial_controller_file = "/Users/javiergutierrez/Documents/Thesis/Thesis_Repo/2D_Docking/fixed_controller_20n_manhattan.pt"
    

    out_controller_folders = "controllers/"
    out_model_folders = "models/"
    out_controller_data_folders = "controller_data/"
    out_comb_folders = "combined/"
    out_dynamic_folders = "dynamics/"
    out_counterexample_folders = "counterexamples/"
    out_dynamic_data_folders = "dynamic_data/"
    
    out_timing_counterexample_file = f"times_counterexamples{lip_ext}.txt"

    
    if not os.path.isdir(out_controller_folders):
        os.mkdir(out_controller_folders)
        os.mkdir(out_model_folders)
        os.mkdir(out_controller_data_folders)
        os.mkdir(out_comb_folders)
        os.mkdir(out_counterexample_folders)
        os.mkdir(out_dynamic_folders)
        os.mkdir(out_dynamic_data_folders)

    cur_controller_data_file = out_controller_data_folders + controller_data_file + str(index) + ".pt"
    cur_controller_data_val_file = out_controller_data_folders + val_file + str(index) + ".pt"
    cur_model_file = out_model_folders + model_file + str(index) + ".pt"
    cur_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
    cur_controller_file = out_controller_folders + controller_file + str(index) + ".pt"
    cur_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
    cur_dynamic_file = out_dynamic_folders + dynamic_file + str(index) + ".pt"
    cur_counterexample_file = out_counterexample_folders + counterexample_file + str(index) + ".pt"
    cur_dynamic_data_file = out_dynamic_data_folders + dynamic_data_file + str(index) + ".pt"

    dynamic_finetune_data_file = out_dynamic_data_folders + f"finetune_data_{lip_ext}.pt"

    dynamic_data_val_file = out_dynamic_data_folders + f"val_{lip_ext}" + ".pt"

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

    error_bound = ((sample_spacing * (dyn_lip + assumed_true_lip)) / 2.0) + max_error   # Add small constant to be safe

    # cert_eps is the epsilon we use to verify the certificate, and the Er in the slide from which the robust epsilon arrises
    #cert_eps = error_bound.detach() + 1e-10
    cert_eps = 0.0000001

    #safe level for verification cerificate
    safe_level = 1.0

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
    train_model(safe_pos, pos_limit, vel_limit, cur_controller_data_file, cur_controller_data_val_file, cur_model_file, cur_controller_file, threshold, initial_controller_file, cur_dynamic_file, lip_loss, descenteps=1e-4)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    f = open(out_timing_counterexample_file, "a")
    print("Total training time for model index", str(index), ":", str(diff.seconds))
    f.write("Total training time for model index " + str(index) + ":" + str(diff.seconds) + "\n")
    f.close()
    # old_mse, _ = stat_eval(cur_dynamic_file, "dynamic_data/train_0.pt")
    # print("dynamic64 mse: ", old_mse)
    combined_model(cur_model_file, cur_controller_file, cur_dynamic_file, cur_comb_file)
    single_model(cur_model_file, cur_model_onnx_file)
    st_ver_time = datetime.now()
    ret, ret_ranges, failed = safe_descent_cond_check(cur_comb_file, cur_model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = vel_limit, descenteps=cert_eps, safe_level=safe_level)
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
        next_controller_data_file = out_controller_data_folders + controller_data_file + str(index) + ".pt"
        next_controller_data_val_file = out_controller_data_folders + val_file + str(index) + ".pt"
        next_model_file = out_model_folders + model_file + str(index) + ".pt"
        next_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
        next_controller_file = out_controller_folders + controller_file + str(index) + ".pt"
        next_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
        next_dynamic_file = out_dynamic_folders + dynamic_file + str(index) + ".pt"
        next_counterexample_file = out_counterexample_folders + counterexample_file + str(index) + ".pt"
        next_dynamic_data_file = out_dynamic_data_folders + dynamic_data_file + str(index) + ".pt"

        counterexamples = parse_counterexamples(cur_dynamic_file, cur_controller_file, ret)
        torch.save(counterexamples, cur_counterexample_file)

        # avg_mse_counterexamples, _ = stat_eval(cur_dynamic_file, cur_counterexample_file)
        # avg_mse_prev, _ = stat_eval(cur_dynamic_file, cur_dynamic_data_file)

        # big_error = 1.05 * avg_mse_prev <= avg_mse_counterexamples

        # # double check dynamic used in controller training
        # if big_error:

        #     f = open(out_timing_counterexample_file, "a")

        #     print(f"Finetuning dynamic: error on counterexamples of {avg_mse_counterexamples}\n")
        #     f.write(f"Finetuning dynamic: error on counterexamples of {avg_mse_counterexamples}\n")


        #     st_train_time = datetime.now()

        #     # Making finetuning data with 0.05 weight of counterexamples to learn
        #     append_training_set(cur_dynamic_data_file, cur_counterexample_file, 0.05, dynamic_finetune_data_file)
        #     finetune_dynamic(cur_dynamic_file, dynamic_finetune_data_file, avg_mse_prev, next_dynamic_file)

        #     # Just one copy of the ocunterexamples in the future dataset
        #     append_training_set(cur_dynamic_data_file, cur_counterexample_file, 0, next_dynamic_data_file)

        #     end_train_time = datetime.now()

        #     diff = end_train_time - st_train_time

        #     print("Total training time for dynamic index", str(index), ":", str(diff.seconds) + "\n")
        #     f.write("Total training time for dynamic index " + str(index) + ":" + str(diff.seconds) + "\n")

        #     mse, mae = stat_eval(cur_dynamic_file, cur_counterexample_file)
        #     print(f"Current dynamic on counterexamples only MSE: {mse:.8}\n")
        #     f.write(f"Current dynamic on counterexamples only MSE: {mse:.8}\n")

        #     mse, mae = stat_eval(cur_dynamic_file, next_dynamic_data_file)
        #     print(f"Current dynamic on next data split MSE: {mse:.8}\n")
        #     f.write(f"Current dynamic on next data split MSE: {mse:.8}\n")

        #     mse, mae = stat_eval(cur_dynamic_file, cur_dynamic_data_file)
        #     print(f"Current dynamic on previous data split MSE: {mse:.8}\n")
        #     f.write(f"Current dynamic on previous data split MSE: {mse:.8}\n")

        #     mse, mae = stat_eval(next_dynamic_file, cur_counterexample_file)
        #     print(f"New dynamic on counterexamples only MSE: {mse:.8}\n")
        #     f.write(f"New dynamic on counterexamples only MSE: {mse:.8}\n")

        #     mse, mae = stat_eval(next_dynamic_file, next_dynamic_data_file)
        #     print(f"New dynamic on next data split MSE: {mse:.8}\n")
        #     f.write(f"New dynamic on next data split MSE: {mse:.8}\n")

        #     mse, mae = stat_eval(next_dynamic_file, cur_dynamic_data_file)
        #     print(f"New dynamic on previous data split MSE: {mse:.8}\n")
        #     f.write(f"New dynamic on previous data split MSE: {mse:.8}\n")
        #     f.close()
        # else:
        #     save_dynamic(cur_dynamic_file, next_dynamic_file)
        #     save_dynamic_data(cur_dynamic_data_file, next_dynamic_data_file)

        save_dynamic(cur_dynamic_file, next_dynamic_file)
        save_dynamic_data(cur_dynamic_data_file, next_dynamic_data_file)

        st_train_time = datetime.now()
        retrain_model(index - 1, torch.Tensor(ret), torch.Tensor(ret_ranges), safe_pos, pos_limit, vel_limit, cur_controller_data_file, next_controller_data_file, next_controller_data_val_file, cur_model_file, cur_controller_file, next_model_file, next_controller_file, next_dynamic_file, threshold, lip_loss, descenteps=1e-4)
        end_train_time = datetime.now()
        diff = end_train_time - st_train_time
        f = open(out_timing_counterexample_file, "a")
        print("Total training time for model index", str(index), ":", str(diff.seconds))
        f.write("Total training time for model index " + str(index) + ":" + str(diff.seconds) + "\n")
        f.close()
        combined_model(next_model_file, next_controller_file, next_dynamic_file, next_comb_file)
        single_model(next_model_file, next_model_onnx_file)
        st_ver_time = datetime.now()
        ret, ret_ranges, failed = safe_descent_cond_check(next_comb_file, next_model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = vel_limit, descenteps=cert_eps, safe_level=safe_level)
        end_ver_time = datetime.now()
        diff_ver_time = end_ver_time - st_ver_time
        f = open(out_timing_counterexample_file, "a")
        print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
        f.write("Total training time for verification index " + str(index) + ":" + str(diff_ver_time.seconds) + "\n")
        f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
        f.write("Number of failed cases: " + str(len(failed)) + "\n")
        f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
        f.close()

        cur_controller_data_file, cur_controller_data_val_file, cur_model_file, cur_controller_file, cur_counterexample_file, cur_dynamic_file, cur_dynamic_data_file = next_controller_data_file, next_controller_data_val_file, next_model_file, next_controller_file, next_counterexample_file, next_dynamic_file, next_dynamic_data_file
    
    print(failed)