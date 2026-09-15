# FedStay: Technical Findings & Performance Report

## Executive Summary
FedStay evaluates privacy-preserving machine learning for healthcare length-of-stay (LOS) prediction. The experiment measures performance trade-offs between centralizing patient data versus aggregating predictions across three isolated hospital nodes.

## Key Empirical Findings
- **Statistical Significance**: Independent t-tests confirmed patient age ($p < 0.001$) and comorbidity counts ($p < 0.001$) as significant predictors of extended stay.
- **Privacy Utility Trade-Off**:
  - **Centralized Model**: Accuracy = ~0.82 | AUC = ~0.86
  - **Federated Model**: Accuracy = ~0.81 | AUC = ~0.85
- **Performance Loss**: Under 2% performance drop when using distributed training without centralizing patient datasets.

## Conclusion
Federated ensemble strategies match centralized baseline performance within a minor margin while providing raw data privacy across local healthcare environments.