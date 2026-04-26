# Secure Federated Learning
This Work extends federated learning with fairness-aware optimization by integrating secure FL into the Rényi-based framework.

Clients train models locally and compute fairness-related statistics. These updates are protected through secure FL layer using key-sharing mechanisms, ensuring that individual client contributions remain hidden. The server then aggregates the masked updates, and updates a global fairness vector.

This design enables a unified analysis of privacy, security, and fairness tradeoffs in federated learning.
```mermaid
graph LR
    Clients --> SecureFL --> Server --> Analysis
```

## Full Pipeline

<details>
<summary><b>Click to expand full system flow</b></summary>

<br>

```mermaid
graph TD

%% ================= CLIENT SIDE =================
subgraph Clients
    C1[Download Global Model and v]
    C2[Local Training]
    C3[Compute Predictions]
    C4[Compute Stats j_bar u_bar]
    C5[Build Fairness Vector x_k]
end

%% ================= SECURE LAYER =================
subgraph SecureFL
    S1[Diffie Hellman Key Generation]
    S2[Shared Secret]
    S3[PRG Mask]
    S4[Mask Local Vector]
end

%% ================= SERVER =================
subgraph Server
    SV1[Aggregate Updates]
    SV2[Add DP Noise]
    SV3[Compute Q_hat]
    SV4[SVD Update v]
end

%% ================= ANALYSIS =================
subgraph Analysis
    A1[Compute Fairness]
    A2[Compare Baseline and DP]
    A3[Analyze Tradeoff]
end

%% ================= FLOW =================
C1 --> C2 --> C3 --> C4 --> C5

%% baseline path
C5 --> SV1

%% secure path
C5 --> S1 --> S2 --> S3 --> S4 --> SV1

%% server pipeline
SV1 --> SV2 --> SV3 --> SV4

%% feedback loop
SV4 --> C1

%% analysis
SV2 --> A1 --> A2 --> A3

%% ================= STYLING =================
style Clients fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
style SecureFL fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
style Server fill:#f5f5f5,stroke:#424242,stroke-width:2px
style Analysis fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

</details>

## Key Idea

* Clients train locally without sharing raw data
* Secure aggregation hides individual updates
* Fairness without compromised is analyzed across clients

## Functioning And Methodology

## 📊 Results and Analysis

Each client computes a local fairness vector containing statistics such as ( j_{c,p} ) and ( u_c ). Before sending this to the server, the vector is masked using values generated from shared cryptographic keys. For example, a value like ( x_k = 183475 ) is transformed into ( y_k = 183475 + 4294660347 = 4294843822 ), making it appear random to the server. Importantly, these masks are constructed so that they cancel out across clients. As a result, when the server aggregates all received values, the masks sum to zero and the true global sum is recovered.

This is confirmed in the aggregation results, where the difference between the baseline (no masking) and secure aggregation is on the order of ( 10^{-6} ), which is negligible and only due to floating-point precision. This demonstrates that secure aggregation preserves correctness while ensuring privacy.

From the aggregated statistics, fairness is computed using the difference in prediction rates across sensitive groups:
( DEO = |P(\hat{y}=1|s=0) - P(\hat{y}=1|s=1)| ), and ( FR = 1 - DEO ).
The final results show an accuracy of 0.8388 and fairness of 0.7622, with a harmonic mean of 0.7987, indicating a strong balance between performance and fairness. Overall, the system successfully ensures that individual client data remains hidden while still enabling accurate and fair global learning.


### 🎯 Key Insights

* Secure aggregation preserves correctness while ensuring privacy
* Individual client contributions remain hidden
* Fairness metrics can still be computed accurately
* Enables joint analysis of:

  * security (masking)
  * privacy (DP)
  * fairness (Rényi framework)

---


## Note

Dataset is not included due to size. Please place it manually in the `dataset/` folder.
