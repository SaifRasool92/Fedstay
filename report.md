# FedStay: Technical Findings & Performance Report

## Executive Summary
FedStay provides a simulated federated learning proof-of-concept for healthcare length-of-stay (LOS) prediction. The experiment measures performance trade-offs between centralizing patient data versus aggregating parameters across three simulated hospital nodes. Note that this is a single-process simulation and does not implement network isolation or secure aggregation.

## Key Empirical Findings
- **Statistical Significance**: Independent t-tests confirmed patient age ($p = 8.43 \times 10^{-26}$) and comorbidity counts ($p = 4.12 \times 10^{-28}$) as significant predictors of extended stay in the synthetic dataset.
- **Privacy Utility Trade-Off**:
  - **Centralized Logistic Regression**: AUC = 0.8143
  - **Centralized Random Forest Baseline**: Accuracy = 0.6933 | AUC = 0.7654
  - **Federated Logistic Regression**: Accuracy = 0.7500 | AUC = 0.8152
- **Performance Loss**: The federated simulation achieved an AUC (0.8152) essentially identical to the centralized Logistic Regression (0.8143).

## Conclusion
A simulated federated Logistic Regression approach using weighted parameter averaging matches centralized baseline performance. Further work is required to transition this simulation into a true privacy-preserving deployment.