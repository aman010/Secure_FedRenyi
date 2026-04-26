import torch
import numpy as np
import copy
import hashlib
import random
import pandas as pd
import sys
import os


# from tool.logger import *
# from tool.utils import get_parameters, set_parameters
from FR.tool.logger import *
from FR.tool.utils import get_parameters, set_parameters


device = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment


def add_laplace_noise(X, epsilon, sensitivity):
    scale = sensitivity / epsilon

    noise = torch.distributions.Laplace(
        torch.tensor(0.0, device=X.device),
        torch.tensor(scale, device=X.device)
    ).sample(X.shape)

    return X + noise, noise

def compute_fairness_from_X(X):
    eps = 1e-8

    j_c0_p0, j_c0_p1, j_c1_p0, j_c1_p1 = X[:4]

    # P(y_hat=1 | s=1)
    x1 = j_c1_p1 / (j_c1_p1 + j_c0_p1 + eps)

    # P(y_hat=1 | s=0)
    x2 = j_c1_p0 / (j_c1_p0 + j_c0_p0 + eps)

    deo = torch.abs(x2 - x1)
    fr = 1 - deo

    return fr.item()


def client_selection(client_num, fraction, dataset_size, client_dataset_size_list, drop_rate, style="FedAvg"):
    assert sum(client_dataset_size_list) == dataset_size

    selected_num = max(int(fraction * client_num), 1)

    if float(drop_rate) != 0:
        drop_num = max(int(selected_num * drop_rate), 1)
        selected_num = max(selected_num - drop_num, 1) 

    probs = np.array(client_dataset_size_list, dtype=float)
    probs = probs / probs.sum()

    if style == "FedAvg":
        idxs_users = np.random.choice(
            a=client_num,  
            size=selected_num,
            replace=False,
            p=probs
        )
    else:
        idxs_users = np.random.choice(
            a=client_num,
            size=selected_num,
            replace=False
        )

    return idxs_users.tolist()   


def r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p):
    r_bar_k_p_list = []
    for k in range(num_clients_K):
        if mask_s1_flag:
            # Sensitive attribute 2
            client_s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[k].indices])
        else:
            # Sensitive attribute 1
            client_s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[k].indices])
        r_bar_k_p = sum(client_s == p) / len(client_s)
        r_bar_k_p_list.append(r_bar_k_p)

    r_bar_k_p_list = np.array(r_bar_k_p_list)
    r_hat_p = r_bar_k_p_list.mean()
    return r_hat_p



# -----------------------------
# Diffie-Hellman Setup
# -----------------------------
P = 2**127 - 1   # large prime (Mersenne prime)
G = 5            # generator

def modexp(base, exp, mod):
    return pow(base, exp, mod)

# -----------------------------
# Key Generation
# -----------------------------
def generate_keys():
    sk = random.randint(1, P-2)
    pk = modexp(G, sk, P)
    return sk, pk

# -----------------------------
# Shared Secret (DH)
# -----------------------------
def compute_shared_secret(sk, pk_other):
    shared = modexp(pk_other, sk, P)
    return shared

# -----------------------------
# Hash → Seed
# -----------------------------
def hash_to_seed(val):
    h = hashlib.sha256(str(val).encode()).digest()
    return int.from_bytes(h[:8], 'big')   # 64-bit seed

# -----------------------------
# PRG (Deterministic)
# -----------------------------
def PRG(seed, dim):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(dim, generator=g)

# -----------------------------
# Build fairness vector
# -----------------------------
def build_fairness_vector(j_vals, u_vals, r_vals=None):
    vec = j_vals + u_vals
    if r_vals is not None:
        vec += r_vals
    return torch.tensor(vec, dtype=torch.float32)

# ----------------------------
def secure_aggregate(client_vectors, M=2**32, SCALE=1e6, debug=True):

    K = len(client_vectors)
    device = client_vectors[0].device
    dim = client_vectors[0].shape[0]

    param_names = [
        "j_c0_p0", "j_c0_p1",
        "j_c1_p0", "j_c1_p1",
        "u_c0", "u_c1"
    ]

    # -------------------------
    # Scale inputs
    # -------------------------
    client_vectors_scaled = [(x * SCALE).long() for x in client_vectors]

    # -------------------------
    # Generate keys
    # -------------------------
    sk_list, pk_list = [], []
    for _ in range(K):
        sk, pk = generate_keys()
        sk_list.append(sk)
        pk_list.append(pk)

    masked_vectors = []
    client_rows = []

    # -------------------------
    # Masking
    # -------------------------
    for k in range(K):
        mask = torch.zeros(dim, dtype=torch.long, device=device)

        for v in range(K):
            if v == k:
                continue

            shared = compute_shared_secret(sk_list[k], pk_list[v])
            seed = hash_to_seed(shared)

            p_kv = PRG(seed, dim).float()
            p_kv = (p_kv * 1e5).long() % M

            if k < v:
                mask = (mask + p_kv) % M
            else:
                mask = (mask - p_kv) % M

        y_k = (client_vectors_scaled[k] + mask) % M
        masked_vectors.append(y_k)

        # -------------------------
        # Logging row
        # -------------------------
        v = (k + 1) % K   
        row = {"Client": f"{k}"}

        x_vals = client_vectors_scaled[k].cpu().tolist()
        mask_vals = mask.cpu().tolist()
        y_vals = y_k.cpu().tolist()

        for i, name in enumerate(param_names):
            row[f"x_{name}"] = x_vals[i]
            row[f"mask_{name}"] = mask_vals[i]
            row[f"y_{name}"] = y_vals[i]

        client_rows.append(row)

    # -------------------------
    # Pretty logs
    # -------------------------
    if debug:
        client_df = pd.DataFrame(client_rows)

        print("\n========= Parameter Logs (No Private Mask) =========")

        for _, row in client_df.iterrows():
            print(f"\n--- CLIENT {row['Client']} ---")

            pretty_rows = []
            for name in param_names:
                pretty_rows.append({
                    "Parameter": name,
                    "x_k": row[f"x_{name}"],
                    "mask": row[f"mask_{name}"],
                    "y_k": row[f"y_{name}"],
                })

            print(pd.DataFrame(pretty_rows).to_string(index=False))

        print("\n==================================================\n")

    # -------------------------
    # Aggregate
    # -------------------------
    Y = torch.stack(masked_vectors).sum(dim=0) % M
    X_secure = Y.float() / SCALE

    # -------------------------
    # Baseline (for verification)
    # -------------------------
    if debug:
        X_base = torch.stack(client_vectors).sum(dim=0)

        diff = (X_base.cpu() - X_secure.cpu())

        agg_df = pd.DataFrame({
            "Dimension": list(range(len(X_base))),
            "Baseline": X_base.cpu().tolist(),
            "Secure": X_secure.cpu().tolist(),
            "Difference": diff.tolist()
        })

        print("\n========= AGGREGATION COMPARISON =========")
        print(agg_df.to_string(index=False))
        print("\nMax diff :", diff.abs().max().item())
        print("=========================================\n")

    return X_secure

    # # DEBUG aggregation
    # X_base = torch.stack(client_vectors).sum(dim=0)
    
    # print("\n--- AGG CHECK ---")
    # print("Base sum (first 5):", X_base[:5])
    # print("Secure sum (first 5):", X[:5])
    # print("Diff:", (X_base - X).abs().max())
    # print("------------------\n")



def q_c_p(y_hat_θ, s, c, p):
    y_hat_θ_c = (y_hat_θ == c).to(device)
    s_p = (s == p).to(device)
    joint = (y_hat_θ_c * s_p).to(device)

    P_y_hat_θ_c = (sum(y_hat_θ_c) / len(y_hat_θ)).to(device)  # u^bar_k(c)
    P_s_p = (sum(s_p) / len(s)).to(device)  # r^bar_k(p)

    P_joint = (sum(joint) / len(s)).to(device)
    P_conditional = (P_joint / P_s_p).to(device)  # j^bar_k(c, p)

    if P_conditional == 0 or P_s_p == 0 or P_y_hat_θ_c == 0:
        q = torch.tensor(0.)
    else:
        q = P_conditional * P_s_p / torch.sqrt(P_y_hat_θ_c * P_s_p)

    return q.to(device)


def G_θ_v(y_hat_θ, s, v):
    q_00 = q_c_p(y_hat_θ, s, 0, 0)
    q_01 = q_c_p(y_hat_θ, s, 0, 1)
    q_10 = q_c_p(y_hat_θ, s, 1, 0)
    q_11 = q_c_p(y_hat_θ, s, 1, 1)
    Q = torch.tensor([
        [q_00, q_01],
        [q_10, q_11]
    ]).to(device)
    v = v.reshape(-1, 1).to(device)
    g = v.T.matmul(Q.T).matmul(Q).matmul(v)
    g = g[0][0].to(device)
    return g


def argmax_v_LR(idxs_users, local_model_list, mask_s1_flag, training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, γ_k_style):
    training_dataset_size = len(training_dataset)

    j_bar_c0_p0_list = []
    j_bar_c0_p1_list = []
    j_bar_c1_p0_list = []
    j_bar_c1_p1_list = []
    u_bar_c0_list = []
    u_bar_c1_list = []

    for id in idxs_users:
        selected_model = local_model_list[id].to(device)

        if mask_s1_flag:
            # Sensitive attribute 2
            client_s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[id].indices]).to(device)
        else:
            # Sensitive attribute 1
            client_s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[id].indices])
        client_X = torch.tensor(np.array([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices])).to(device)

        y_hat_θ = (selected_model(client_X) >= 0.5).reshape(-1).to(device)

        y_hat_θ_c0 = (y_hat_θ == 0).to(device)
        y_hat_θ_c1 = (y_hat_θ == 1).to(device)

        s_p0 = (client_s == 0).to(device)
        s_p1 = (client_s == 1).to(device)

        joint_c0_p0 = (y_hat_θ_c0 * s_p0).to(device)
        joint_c0_p1 = (y_hat_θ_c0 * s_p1).to(device)
        joint_c1_p0 = (y_hat_θ_c1 * s_p0).to(device)
        joint_c1_p1 = (y_hat_θ_c1 * s_p1).to(device)

        P_s_p0 = (sum(s_p0) / len(client_s)).to(device)
        P_s_p1 = (sum(s_p1) / len(client_s)).to(device)

        P_joint_c0_p0 = (sum(joint_c0_p0) / len(client_s)).to(device)
        P_joint_c0_p1 = (sum(joint_c0_p1) / len(client_s)).to(device)
        P_joint_c1_p0 = (sum(joint_c1_p0) / len(client_s)).to(device)
        P_joint_c1_p1 = (sum(joint_c1_p1) / len(client_s)).to(device)

        if "uniform_client" in γ_k_style:
            γ_k = 1 / len(idxs_users)
        else:
            γ_k = len(client_X) / training_dataset_size

        j_bar_c0_p0 = (γ_k * P_joint_c0_p0 / P_s_p0).to(device)
        j_bar_c0_p1 = (γ_k * P_joint_c0_p1 / P_s_p1).to(device)
        j_bar_c1_p0 = (γ_k * P_joint_c1_p0 / P_s_p0).to(device)
        j_bar_c1_p1 = (γ_k * P_joint_c1_p1 / P_s_p1).to(device)

        j_bar_c0_p0_list.append(j_bar_c0_p0)
        j_bar_c0_p1_list.append(j_bar_c0_p1)
        j_bar_c1_p0_list.append(j_bar_c1_p0)
        j_bar_c1_p1_list.append(j_bar_c1_p1)

        u_bar_c0 = (γ_k * sum(y_hat_θ_c0) / len(y_hat_θ)).to(device)
        u_bar_c1 = (γ_k * sum(y_hat_θ_c1) / len(y_hat_θ)).to(device)  # u^bar_k(c)
        u_bar_c0_list.append(u_bar_c0)
        u_bar_c1_list.append(u_bar_c1)


    j_hat_c0_p0 = torch.tensor(j_bar_c0_p0_list).sum()
    j_hat_c0_p1 = torch.tensor(j_bar_c0_p1_list).sum()
    j_hat_c1_p0 = torch.tensor(j_bar_c1_p0_list).sum()
    j_hat_c1_p1 = torch.tensor(j_bar_c1_p1_list).sum()
    u_hat_c0 = torch.tensor(u_bar_c0_list).sum()
    u_hat_c1 = torch.tensor(u_bar_c1_list).sum()

    q_hat_00 = j_hat_c0_p0 * r_hat_p0 / torch.sqrt(u_hat_c0 * r_hat_p0)
    q_hat_01 = j_hat_c0_p1 * r_hat_p1 / torch.sqrt(u_hat_c0 * r_hat_p1)
    q_hat_10 = j_hat_c1_p0 * r_hat_p0 / torch.sqrt(u_hat_c1 * r_hat_p0)
    q_hat_11 = j_hat_c1_p1 * r_hat_p1 / torch.sqrt(u_hat_c1 * r_hat_p1)
    Q_hat = torch.tensor([
        [q_hat_00, q_hat_01],
        [q_hat_10, q_hat_11]
    ]).to(device)
    logger.info("===== Q_hat =====")
    logger.info(
        f"""
        [{q_hat_00.item():.4f}, {q_hat_01.item():.4f}]
        [{q_hat_10.item():.4f}, {q_hat_11.item():.4f}]
        """
    )

    u, s, v = torch.svd(Q_hat)

    second_singular_vector_of_Q_hat = v[1].reshape(-1, 1).to(device)
    return second_singular_vector_of_Q_hat




def argmax_v_NN(
    idxs_users,
    local_model_list,
    mask_s1_flag,
    training_dataset,
    client_dataset_list,
    r_hat_p0,
    r_hat_p1,
    γ_k_style,
    round_id,
    use_secure=True,
    return_logs=False,
    use_dp = False,
    epsilon = None

):
    device = r_hat_p0.device
    eps = 1e-8

    training_dataset_size = len(training_dataset)

    j_bar_c0_p0_list = []
    j_bar_c0_p1_list = []
    j_bar_c1_p0_list = []
    j_bar_c1_p1_list = []
    u_bar_c0_list = []
    u_bar_c1_list = []

    # -------------------------
    # STEP 1: Collect stats
    # -------------------------
    for id in idxs_users:
        model = local_model_list[id]

        if mask_s1_flag:
            client_s = torch.tensor(
                [training_dataset[idx]['s2'] for idx in client_dataset_list[id].indices],
                device=device
            )
        else:
            client_s = torch.tensor(
                [training_dataset[idx]['s1'] for idx in client_dataset_list[id].indices],
                device=device
            )

        client_X = torch.tensor(
            np.array([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices]),
            device=device
        )

        y_hat = model(client_X).argmax(dim=1)

        y_c0 = (y_hat == 0).float()
        y_c1 = (y_hat == 1).float()

        s_p0 = (client_s == 0).float()
        s_p1 = (client_s == 1).float()

        joint_c0_p0 = y_c0 * s_p0
        joint_c0_p1 = y_c0 * s_p1
        joint_c1_p0 = y_c1 * s_p0
        joint_c1_p1 = y_c1 * s_p1

        P_s_p0 = s_p0.sum() / len(client_s)
        P_s_p1 = s_p1.sum() / len(client_s)

        P_joint_c0_p0 = joint_c0_p0.sum() / len(client_s)
        P_joint_c0_p1 = joint_c0_p1.sum() / len(client_s)
        P_joint_c1_p0 = joint_c1_p0.sum() / len(client_s)
        P_joint_c1_p1 = joint_c1_p1.sum() / len(client_s)

        if "uniform_client" in γ_k_style:
            γ_k = 1 / len(idxs_users)
        else:
            γ_k = len(client_X) / training_dataset_size


        # safe division
        j_bar_c0_p0_list.append(γ_k * P_joint_c0_p0 / (P_s_p0 + eps))
        j_bar_c0_p1_list.append(γ_k * P_joint_c0_p1 / (P_s_p1 + eps))
        j_bar_c1_p0_list.append(γ_k * P_joint_c1_p0 / (P_s_p0 + eps))
        j_bar_c1_p1_list.append(γ_k * P_joint_c1_p1 / (P_s_p1 + eps))

        u_bar_c0_list.append(γ_k * y_c0.sum() / len(y_hat))
        u_bar_c1_list.append(γ_k * y_c1.sum() / len(y_hat))

    # -------------------------
    # STEP 2: Build vectors
    # -------------------------
    x_list = []
    for i in range(len(j_bar_c0_p0_list)):
        x_k = build_fairness_vector(
            [
                j_bar_c0_p0_list[i],
                j_bar_c0_p1_list[i],
                j_bar_c1_p0_list[i],
                j_bar_c1_p1_list[i],
            ],
            [
                u_bar_c0_list[i],
                u_bar_c1_list[i],
            ],
        )
        x_list.append(x_k)

    # -------------------------
    # STEP 3: Aggregation (BOTH)
    # -------------------------
    # X_secure = secure_aggregate(x_list)
    # X_base = torch.stack(x_list).sum(dim=0)
    X_base = torch.stack(x_list).sum(dim=0)
    if use_secure:
        X_secure = secure_aggregate(x_list)
    # -------------------------
    # DEBUG PRINT
    # -------------------------
    if use_secure and return_logs:
        print(f"\n========= AGGREGATION COMPARISON | Round {round_id} =========")
    
        X_base_cpu = X_base.cpu()
        X_secure_cpu = X_secure.cpu()
        diff = (X_base_cpu - X_secure_cpu)
        
        agg_df = pd.DataFrame({
            "Dimension": list(range(len(X_base_cpu))),
            "Baseline": X_base_cpu.tolist(),
            "Secure": X_secure_cpu.tolist(),
            "Difference": diff.tolist()
        })
        
        print(agg_df.to_string(index=False))
        print("\nMax diff :", diff.abs().max().item())
        print("=========================================\n")

    # -------------------------
    # STEP 4: Choose aggregation
    # -------------------------
    # For debugging → keep baseline
    noise = 0.0
    X = X_secure if use_secure else X_base
    if use_dp:
        sensitivity = 1.0 / training_dataset_size
        scale = sensitivity /epsilon
        X, noise = add_laplace_noise(X, epsilon, sensitivity)
        noise = noise
    # Later switch to:
    # X = X_secure
    
    # unpack
    j_hat_c0_p0, j_hat_c0_p1, j_hat_c1_p0, j_hat_c1_p1 = X[:4]
    u_hat_c0, u_hat_c1 = X[4:]

    # -------------------------
    # STEP 5: Compute Q_hat
    # -------------------------
    q_hat_00 = j_hat_c0_p0 * r_hat_p0 / torch.sqrt(u_hat_c0 * r_hat_p0 + eps)
    q_hat_01 = j_hat_c0_p1 * r_hat_p1 / torch.sqrt(u_hat_c0 * r_hat_p1 + eps)
    q_hat_10 = j_hat_c1_p0 * r_hat_p0 / torch.sqrt(u_hat_c1 * r_hat_p0 + eps)
    q_hat_11 = j_hat_c1_p1 * r_hat_p1 / torch.sqrt(u_hat_c1 * r_hat_p1 + eps)

    Q_hat = torch.stack([
        torch.stack([q_hat_00, q_hat_01]),
        torch.stack([q_hat_10, q_hat_11])
    ]).to(device)

    # SVD
    _, _, v = torch.svd(Q_hat)

    # -------------------------
    # STEP 6: Logging
    # -------------------------
    if use_dp:
        print(noise)
    log_dict = {
        "j_hat": X[:4],
        "u_hat": X[4:],
        "r_hat": [r_hat_p0.item(), r_hat_p1.item()],
        "mode": "secure" if use_secure else "baseline",
        "noise": noise
    }

    if return_logs:
        return v[1].reshape(-1, 1), log_dict
    else:
        return v[1].reshape(-1, 1)
        
def Fed_Renyi_LR(device,
                 mask_s1_flag,
                 lamda,
                 global_model,
                 algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate, local_step_size,
                 training_dataloaders,
                 training_dataset,
                 client_dataset_list,
                 γ_k_style
                 ):
    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    # global_v_0 = global_v_0_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list)
    global_v = torch.rand(2, 1)

    r_hat_p0 = r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=0)
    r_hat_p1 = r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=1)

    criterion = torch.nn.BCELoss()

    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_K):
            model = local_model_list[i]
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=local_step_size)
            client_i_dataloader = training_dataloaders[i]

            # local option
            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_K};  ##########")
            for batch_index, batch in enumerate(client_i_dataloader):
                X = batch["X"].to(device)
                y = batch["y"].reshape(-1, 1).to(device)
                if mask_s1_flag:
                    s = batch["s2"]
                else:
                    s = batch["s1"]

                local_prediction = model(X).to(device)
                loss = criterion(local_prediction, y.float())
                y_hat_θ = (local_prediction >= 0.5).reshape(-1).to(device)
                regularization_term = lamda * G_θ_v(y_hat_θ, s, global_v).to(device)
                regularization_term = torch.where(torch.isnan(regularization_term), torch.full_like(regularization_term, 0), regularization_term)

                if torch.isnan(regularization_term):
                    logger.info("Regularization term is nan, now fix it to 0.")
                if batch_index % 10 == 0:
                    logger.info(f"      @@@@ "
                                f"Batch: {batch_index}; "
                                f"Cross Entropy Loss: {round(float(loss),4)}; "
                                f"Regularization term: {round(float(regularization_term),4)}; "
                                f"Total Loss: {round(float(loss+regularization_term),4)}; "
                                f"@@@@")

                loss += regularization_term
                loss.backward()
                optimizer.step()

            # Upgrade the local model list
            local_model_list[i] = model

        # Communicate
        if (iter_t + 1) % communication_round_I == 0:
            logger.info(f"********** Communicate: {(iter_t + 1) / communication_round_I} **********")
            # Client selection
            idxs_users = client_selection(
                client_num=num_clients_K,
                fraction=FL_fraction,
                dataset_size=training_dataset_size,
                client_dataset_size_list=client_datasets_size_list,
                drop_rate=FL_drop_rate,
                style="FedAvg",
            )
            logger.info(f"Select client list: {idxs_users} ")

            # Global operation
            theta_list = []
            for id in idxs_users:
                selected_model = local_model_list[id]
                if "uniform_client" in γ_k_style:
                    γ_k = 1 / len(idxs_users)
                else:
                    γ_k = len([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices]) / training_dataset_size

                theta_list.append(list(γ_k * np.array(get_parameters(selected_model))))

            logger.info("********** Parameter aggregation **********")
            theta_list = np.array(theta_list, dtype=object)
            theta_avg = np.sum(theta_list, 0).tolist()
            set_parameters(global_model, theta_avg)

            logger.info("********** Global v update **********")
            backup_v = global_v

            try:
                global_v = argmax_v_LR(idxs_users, local_model_list, mask_s1_flag, training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, γ_k_style)
            except Exception:
                global_v = backup_v
            # Parameter Distribution
            logger.info("********** Parameter distribution **********")
            local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    logger.info("Training finish, return global model")
    return global_model


    
def Fed_Renyi_NN(
    device,
    mask_s1_flag,
    lamda,
    global_model,
    algorithm_epoch_T,
    num_clients_K,
    communication_round_I,
    FL_fraction,
    FL_drop_rate,
    local_step_size,
    training_dataloaders,
    training_dataset,
    client_dataset_list,
    γ_k_style,
    return_logs=False,
    use_dp = False,
    epsilons = None
):
    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    dp_results = {
    "epsilon": [],
    "fairness": [],
    "baseline": [],
    "rounds": [],
     "noise": []
                 }

    logger.info("Training process")

    # -------------------------
    # Initialization
    # -------------------------
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]
    global_v = torch.rand(2, 1).to(device)

    r_hat_p0 = r_hat_p_initialization(
        num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=0
    )
    r_hat_p1 = r_hat_p_initialization(
        num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=1
    )

    criterion = torch.nn.CrossEntropyLoss()

    # =========================
    # TRAINING LOOP
    # =========================
    for iter_t in range(algorithm_epoch_T):

        # -------------------------
        # LOCAL TRAINING
        # -------------------------
        for i in range(num_clients_K):
            model = local_model_list[i]
            model.train()

            optimizer = torch.optim.SGD(model.parameters(), lr=local_step_size)
            client_loader = training_dataloaders[i]

            logger.info(
                f"########## Epoch {iter_t+1}/{algorithm_epoch_T} | Client {i+1}/{num_clients_K} ##########"
            )

            for batch in client_loader:
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                s = batch["s2"] if mask_s1_flag else batch["s1"]

                pred = model(X)
                loss = criterion(pred, y.long())

                y_hat = torch.argmax(pred, dim=1)

                reg = lamda * G_θ_v(y_hat, s, global_v)
                reg = torch.where(torch.isnan(reg), torch.zeros_like(reg), reg)

                loss = loss + reg
                loss.backward()
                optimizer.step()

        # -------------------------
        # COMMUNICATION
        # -------------------------
        if (iter_t + 1) % communication_round_I == 0:

            logger.info(f"********** Communication Round {(iter_t+1)//communication_round_I} **********")

            idxs_users = client_selection(
                client_num=num_clients_K,
                fraction=FL_fraction,
                dataset_size=training_dataset_size,
                client_dataset_size_list=client_datasets_size_list,
                drop_rate=FL_drop_rate,
                style="FedAvg",
            )

            logger.info(f"Selected clients: {idxs_users}")

            # -------------------------
            # MODEL AGGREGATION
            # -------------------------
            theta_list = []
            for id in idxs_users:
                model = local_model_list[id]

                if "uniform_client" in γ_k_style:
                    γ_k = 1 / len(idxs_users)
                else:
                    γ_k = len(client_dataset_list[id]) / training_dataset_size

                params = get_parameters(model)
                scaled = [γ_k * p for p in params]
                theta_list.append(scaled)

            theta_avg = np.sum(np.array(theta_list, dtype=object), axis=0).tolist()
            set_parameters(global_model, theta_avg)

            # -------------------------
            # FAIRNESS UPDATE + SECURE CHECK
            # -------------------------
            logger.info("********** Global v update **********")

            for m in local_model_list:
                m.eval()

            with torch.no_grad():

                # BASELINE
                v_base, logs_base = argmax_v_NN(
                    idxs_users,
                    local_model_list,
                    mask_s1_flag,
                    training_dataset,
                    client_dataset_list,
                    r_hat_p0,
                    r_hat_p1,
                    γ_k_style,
                    round_id  = (iter_t + 1) // communication_round_I,
                    use_secure=False,
                    return_logs=True,
                    epsilon = None,
                    use_dp = False
                
                )
                if use_dp:
                    print(f"\n===== BaseLine=====")
                    X_dp = torch.cat([logs_base["j_hat"], logs_base["u_hat"]])
                    fairness = compute_fairness_from_X(X_dp)
                    print("Fairness: {}".format(fairness))
                    print("j_hat:", logs_base["j_hat"].cpu().numpy())
                    print("u_hat:", logs_base["u_hat"].cpu().numpy())
                    print("r_hat:", logs_base["r_hat"])
                    print("Mode: BaseLine")
                    dp_results["baseline"].append(fairness)
                    dp_results["rounds"].append((iter_t + 1) // communication_round_I)
                    print("=================================")
                
                if not use_dp:
                    # SECURE (only for comparison)
                    v_sec, logs_secure = argmax_v_NN(
                        idxs_users,
                        local_model_list,
                        mask_s1_flag,
                        training_dataset,
                        client_dataset_list,
                        r_hat_p0,
                        r_hat_p1,
                        γ_k_style,
                        round_id = (iter_t + 1) // communication_round_I,
                        use_secure=True,
                        return_logs=True,
                        epsilon = None,
                        use_dp = False)
                else:
                    # dP only
                    try:
                        for eps in epsilons:
                            v_dp, logs_DP = argmax_v_NN(
                                idxs_users,
                                local_model_list,
                                mask_s1_flag,
                                training_dataset,
                                client_dataset_list,
                                r_hat_p0,
                                r_hat_p1,
                                γ_k_style,
                                round_id = (iter_t + 1) // communication_round_I,
                                use_secure=False,
                                return_logs=True,
                                epsilon = eps,
                                use_dp = True
                            )
                            print(f"\n===== DP LOG | ε={eps} =====")
                            X_dp = torch.cat([logs_DP["j_hat"], logs_DP["u_hat"]])
                            fairness = compute_fairness_from_X(X_dp)
                            print("Fairness: {}".format(fairness))
                            print("j_hat:", logs_DP["j_hat"].cpu().numpy())
                            print("u_hat:", logs_DP["u_hat"].cpu().numpy())
                            print("r_hat:", logs_DP["r_hat"])
                            print("Mode: DP")
                            print("=================================")

                            dp_results["epsilon"].append(eps)
                            dp_results["fairness"].append(fairness)
                            dp_results["rounds"].append((iter_t + 1) // communication_round_I)
                            dp_results["noise"].append(logs_DP["noise"])
    
                    except IndexError as e:
                            print(f"Caught an exception: {e}")
                # ONLY baseline affects training
                global_v = v_base
            # back to train mode
            for m in local_model_list:
                m.train()

            # distribute global model
            local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]

    logger.info("Training finished")
    return global_model, dp_results

        
        