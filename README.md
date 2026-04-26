# Federated Learning with Rényi DP and Secure Aggregation

A system-level study of **privacy–utility–fairness tradeoffs** in federated learning using differential privacy and (optional) secure aggregation.

---

## 📊 System Overview

```mermaid
graph LR
    Clients --> SecureFL --> Server --> Analysis
```

* **Clients**: local training + fairness statistics
* **SecureFL**: key sharing + masking (optional)
* **Server**: aggregation + DP + fairness update
* **Analysis**: fairness vs privacy evaluation

---

## 🔍 Module Details

<details>
<summary><b>Click to explore modules</b></summary>

### 🧠 Clients

```mermaid
graph TD
    C1[Download Global Model and v]
    C2[Local Training]
    C3[Compute Predictions]
    C4[Compute Stats j_bar u_bar]
    C5[Build Fairness Vector x_k]
    C1 --> C2 --> C3 --> C4 --> C5
```

### 🔐 SecureFL (Optional)

```mermaid
graph TD
    S1[Diffie Hellman Key Generation]
    S2[Compute Shared Secret]
    S3[Generate PRG Mask]
    S4[Mask Local Vector]
    S5[Upload Masked Data]
    S1 --> S2 --> S3 --> S4 --> S5
```

### 🖥️ Server + DP

```mermaid
graph TD
    SV1[Aggregate Updates]
    SV2[Add DP Noise]
    SV3[Compute Q_hat]
    SV4[SVD Update Fairness Vector v]
    SV1 --> SV2 --> SV3 --> SV4
```

### 📊 Analysis

```mermaid
graph TD
    A1[Compute Fairness]
    A2[Compare Baseline and DP]
    A3[Analyze Tradeoff]
    A1 --> A2 --> A3
```

</details>

---

## 🔬 Full System Pipeline

<details>
<summary><b>Click to view full pipeline</b></summary>

<br>

```mermaid
graph TD

%% CLIENTS
subgraph Clients
    C1[Download Model] --> C2[Train]
    C2 --> C3[Compute Stats]
    C3 --> C4[Build Vector]
end

%% SECURE FL
subgraph SecureFL
    S1[Key Generation] --> S2[Shared Secret]
    S2 --> S3[Mask]
end

%% SERVER
subgraph Server
    SV1[Aggregate] --> SV2[DP Noise]
    SV2 --> SV3[Compute Q_hat]
    SV3 --> SV4[SVD Update]
end

%% ANALYSIS
subgraph Analysis
    A1[Fairness] --> A2[Compare] --> A3[Tradeoff]
end

%% FLOW
C4 --> SV1
C4 --> S1
S3 --> SV1

SV4 --> C1
SV2 --> A1 --> A2 --> A3
```

</details>

---

## 💡 Key Idea

* Train models **without sharing raw data**
* Optionally **mask updates** via key sharing
* Add **differential privacy noise** at aggregation
* Evaluate **fairness across clients**

---

## 🧪 Results (Typical Observation)

* Increasing noise → stronger privacy
* But → **fairness gap increases** (especially for smaller clients)
* Highlights a **privacy–fairness tradeoff**

---

## ⚠️ Notes

* Dataset is **not included** due to size
* Place data in `dataset/` before running
* Secure aggregation is **optional** in this implementation

---

## 🚀 How to Run

```bash
python main.py
```

---

## 📁 Project Structure

```
.
├── algorithm/
├── dataset/        # (not included)
├── tool/
├── main.py
└── README.md
```

---

## 📌 Contribution

* Integration of **Rényi DP** in federated learning
* Analysis of **fairness under privacy noise**
* Optional **secure aggregation pipeline**
* Clear **system-level decomposition**

---

## 🧠 Takeaway

> Even small privacy noise can disproportionately affect clients, leading to fairness challenges in federated systems.
