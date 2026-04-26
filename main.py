import numpy as np
import torch
import sys
import os

# Path to your project root (the folder that contains BOTH 'FR' and 'tool')
PROJECT_ROOT = os.path.abspath(".")

sys.path.append(PROJECT_ROOT)
# !python -m pip install matplotlib

# import pandas as pd


import os
import sys
repo_path = '/content/Federated-Renyi'
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Fix imports
files_to_patch = [
    './Federated-Renyi/FR/algorithm/FederatedRenyi.py',
    './Federated-Renyi/FR/algorithm/FederatedAverage.py',
    './Federated-Renyi/FR/algorithm/FederatedFair.py',
    './Federated-Renyi/FR/algorithm/LCO.py',
]

if os.path.exists('./Federated-Renyi/FR/algorithm/FederatedRenyi.py'):
    for filepath in files_to_patch:
        os.system(f"sed -i 's/^from tool.logger import \\*/from ..tool.logger import \\*/g' {filepath}")
        os.system(f"sed -i 's/^from tool.utils import/from ..tool.utils import/g' {filepath}")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import pickle
import statistics

from FR.hypothesis.LogisticRegression import RenyiLogisticRegression
from FR.hypothesis.NeuralNetwork import RenyiNeuralNetwork
from FR.moudle.dataset import get_ADULT_dataset, get_COMPAS_dataset, get_DRUG_dataset
from FR.moudle.dataloader import get_FL_dataloader
from FR.algorithm.FederatedRenyi import Fed_Renyi_NN

def Experiment_Create_dataset(param_dict, no_pickle=False):
    dataset_name = param_dict['dataset_name']
    mask_s1_flag = param_dict['mask_s1_flag']
    mask_s2_flag = param_dict['mask_s2_flag']
    mask_s1_s2_flag = param_dict['mask_s1_s2_flag']

    BASE_DATA_PATH = "FR/dataset"

    if "ADULT" in dataset_name:
        pickle_path = os.path.join(BASE_DATA_PATH, "ADULT/ADULT.pickle")
        data_path = os.path.join(BASE_DATA_PATH, "ADULT")
        get_dataset = get_ADULT_dataset

    elif "COMPAS" in dataset_name:
        pickle_path = os.path.join(BASE_DATA_PATH, "COMPAS/COMPAS.pickle")
        data_path = os.path.join(BASE_DATA_PATH, "COMPAS")
        get_dataset = get_COMPAS_dataset

    else:
        pickle_path = os.path.join(BASE_DATA_PATH, "DRUG/DRUG.pickle")
        data_path = os.path.join(BASE_DATA_PATH, "DRUG")
        get_dataset = get_DRUG_dataset

    if not os.path.exists(pickle_path) or no_pickle:
        training_dataset, testing_dataset = get_dataset(
            data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag
        )

        with open(pickle_path, 'wb') as p:
            pickle.dump({
                "training_dataset": training_dataset,
                "testing_dataset": testing_dataset,
            }, p)

    else:
        with open(pickle_path, 'rb') as r:
            pickle_dict = pickle.load(r)

        training_dataset = pickle_dict['training_dataset']
        testing_dataset = pickle_dict['testing_dataset']

    nn_input_size = training_dataset.X.shape[1]
    return training_dataset, testing_dataset, nn_input_size


def Experiment_Create_dataloader(param_dict, training_dataset, testing_dataset):
    training_dataloaders, validation_dataloaders, client_dataset_list = get_FL_dataloader(
        training_dataset,
        param_dict['num_clients_K'],
        split_strategy="Uniform",
        do_train=True,
        need_validation=param_dict['need_validation'],
        batch_size=param_dict['batch_size'],
        num_workers=0,
        do_shuffle=True
    )

    testing_dataloader = get_FL_dataloader(
        testing_dataset,
        param_dict['num_clients_K'],
        split_strategy="Uniform",
        do_train=False,
        batch_size=param_dict['batch_size'],
        num_workers=0
    )

    return training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader


def Experiment_Model_construction(param_dict, nn_input_size):
    if param_dict['hypothesis'] == "LR":
        model = RenyiLogisticRegression(input_size=nn_input_size)
    else:
        model = RenyiNeuralNetwork(input_size=nn_input_size, hidden_size=12)

    model.to(param_dict['device'])
    return model


def Experiment_Model_testing(device, testing_dataloader, mask_s1_flag, global_model, hypothesis):
    acc_numerator = 0
    acc_denominator = 0

    num_s1_pred1 = 0
    num_s1_pred0 = 0
    num_s0_pred1 = 0
    num_s0_pred0 = 0

    for batch in testing_dataloader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        if hypothesis == "LR":
            pred = (global_model(X) >= 0.5).reshape(-1)
        else:
            pred = global_model(X).argmax(dim=1)

        acc_numerator += int(sum(pred.eq(y)))
        acc_denominator += X.shape[0]

        s = batch["s2"].to(device) if mask_s1_flag else batch["s1"].to(device)

        y_1 = (y == 1).int().to(device)
        s_1 = (s == 1).int().to(device)
        s_0 = (s == 0).int().to(device)
        pred_1 = (pred == 1).int().to(device)
        pred_0 = (pred == 0).int().to(device)

        num_s1_pred1 += (y_1 * s_1 * pred_1).sum()
        num_s1_pred0 += (y_1 * s_1 * pred_0).sum()
        num_s0_pred1 += (y_1 * s_0 * pred_1).sum()
        num_s0_pred0 += (y_1 * s_0 * pred_0).sum()

    acc = acc_numerator / acc_denominator
    x1 = num_s1_pred1 / (num_s1_pred1 + num_s1_pred0)
    x2 = num_s0_pred1 / (num_s0_pred1 + num_s0_pred0)
    
    DEO = abs(x2 - x1)
    FR = 1 - DEO
    HM = statistics.harmonic_mean([float(acc), float(FR)])

    return float(acc), float(FR), float(HM)

def dp_sanity_check(model_full, model_removed, test_dl, device, hypothesis):
    diffs = []

    model_full.eval()
    model_removed.eval()

    with torch.no_grad():
        for batch in test_dl:
            X = batch["X"].to(device)

            if hypothesis == "LR":
                out1 = model_full(X)
                out2 = model_removed(X)
            else:
                out1 = torch.softmax(model_full(X), dim=1)
                out2 = torch.softmax(model_removed(X), dim=1)

            diff = torch.mean(torch.abs(out1 - out2)).item()
            diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs)
    print(f"\n[DP SANITY CHECK] Avg output difference: {avg_diff:.6f}")
    return avg_diff


def combine_client_datasets(client_dataset_list):
    X_all, y_all, s1_all, s2_all = [], [], [], []
    new_client_list = []

    current_idx = 0

    for client in client_dataset_list:
        base = client.dataset
        idx = client.indices

        X = torch.tensor(base.X[idx], dtype=torch.float32)
        y = torch.tensor(base.y[idx], dtype=torch.long)
        s1 = torch.tensor(base.s1[idx], dtype=torch.long)
        s2 = torch.tensor(base.s2[idx], dtype=torch.long)

        X_all.append(X)
        y_all.append(y)
        s1_all.append(s1)
        s2_all.append(s2)

        # NEW indices (remapped)
        new_indices = list(range(current_idx, current_idx + len(idx)))

        # Create new Subset-like object
        class TempSubset:
            def __init__(self, indices):
                self.indices = indices
        
            def __len__(self):
                return len(self.indices)

        temp_subset = TempSubset(new_indices)

        new_client_list.append(temp_subset)

        current_idx += len(idx)

    class TempDataset:            
        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return {
                "X": self.X[idx],
                "y": self.y[idx],
                "s1": self.s1[idx],
                "s2": self.s2[idx],
            }

    new_dataset = TempDataset()
    new_dataset.X = torch.cat(X_all, dim=0)
    new_dataset.y = torch.cat(y_all, dim=0)
    new_dataset.s1 = torch.cat(s1_all, dim=0)
    new_dataset.s2 = torch.cat(s2_all, dim=0)

    return new_dataset, new_client_list
    
   

def main(base_path, dataset_name, hypothesis, γ_k_style, device, fL_Fraction, epsilon_list = None):
    param_dict = {}
    results = []

    with open(os.path.join(base_path, "COMMON.json"), "r") as f:
        param_dict.update(json.load(f))
    with open(os.path.join(base_path, dataset_name + ".json"), "r") as f:
        param_dict.update(json.load(f))

    param_dict['γ_k_style'] = γ_k_style
    param_dict['dataset_name'] = dataset_name
    # param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['device'] = "cuda" if ("gpu" in device and torch.cuda.is_available()) else "cpu"


    param_override = {
    'num_clients_K': 5,
    'batch_size': 32,
    'algorithm_epoch_T': 12,
    'communication_round_I': 6,
    'FL_drop_rate': 0.4,
    'lamda': 0.85,
    }
    param_dict.update(param_override)

    

    training_dataset, testing_dataset, nn_input_size = Experiment_Create_dataset(param_dict)
    train_dl, val_dl, client_list, test_dl = Experiment_Create_dataloader(
        param_dict, training_dataset, testing_dataset
    )
    
    
    param_dict.update({"FL_fraction":fL_Fraction})
    

    # -------- FedRenyi --------
    model_renyi = Experiment_Model_construction(param_dict, nn_input_size)
    
    model_renyi, dp_res = Fed_Renyi_NN(
    param_dict['device'],
    param_dict['mask_s1_flag'],
    param_dict['lamda'],
    model_renyi,
    param_dict['algorithm_epoch_T'],
    param_dict['num_clients_K'],
    param_dict['communication_round_I'],
    param_dict['FL_fraction'],
    param_dict['FL_drop_rate'],
    param_dict['local_step_size'],
    train_dl,
    training_dataset,
    client_list,
    γ_k_style,
    return_logs=True,
    use_dp = False,
    epsilons = epsilon_list
)


    
    acc, fr, hm = Experiment_Model_testing(
        param_dict['device'], test_dl, param_dict['mask_s1_flag'], model_renyi, hypothesis
    )

   
    results.append(("FedRenyi", acc, fr, hm))
    print("\n===== RESULTS =====")
    print(f"{'Algorithm':<12} {'Accuracy':<10} {'Fairness':<10} {'HM':<10}")
    
    for name, acc, fr, hm in results:
        print(f"{name:<12} {acc:.4f}     {fr:.4f}     {hm:.4f}")
    return results, dp_res

PATH = "./"


