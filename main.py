import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# Suppress non-critical convergence or runtime warnings for clean terminal presentation
warnings.filterwarnings('ignore')

def setup_environment():
    print("....Creating Directory Structure....")
    os.makedirs("data/hospital_sites", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)

def generate_and_split_data():
    print("....Generating Synthetic Healthcare Data....")
    np.random.seed(42)
    n_samples = 1500
    ages = np.random.randint(18, 90, n_samples)
    admission_types = np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples, p=[0.5, 0.3, 0.2])
    diagnosis_cats = np.random.choice(['Cardiology', 'Oncology', 'Neurology', 'Orthopedics'], n_samples)
    prior_visits = np.random.poisson(lam=1.5, size=n_samples)
    comorbidities = np.random.poisson(lam=1.2, size=n_samples)

    los_base = 2 + (ages * 0.05) + (prior_visits * 0.8) + (comorbidities * 1.2)
    los_base += np.where(admission_types == 'Emergency', 2.5, 0)
    los_days = np.clip(np.round(los_base + np.random.normal(0, 2, n_samples)), 1, 30)

    df = pd.DataFrame({
        'age': ages,
        'admission_type': admission_types,
        'diagnosis_category': diagnosis_cats,
        'prior_visits': prior_visits,
        'comorbidity_count': comorbidities,
        'los_days': los_days
    })
    median_los = df['los_days'].median()
    df['long_stay'] = (df['los_days'] > median_los).astype(int)
    df.to_csv("data/los_data.csv", index=False)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['long_stay'])

    index_splits = np.array_split(train_df.index, 3)
    print("\n....Federated Site Distribution (Train Only)....")
    for i, idxs in enumerate(index_splits):
        site_df = train_df.loc[idxs]
        site_df.to_csv(f"data/hospital_sites/hospital_site_{i+1}.csv", index=False)
        pos_ratio = site_df['long_stay'].mean()
        print(f"Site {i+1} -> Samples: {len(site_df)} | Long Stay Ratio: {pos_ratio:.2%}")

    return train_df, test_df

def run_stats_and_eda(df):
    print("\n....Running Statistical Tests & Heatmap....")
    num_cols = ['age', 'prior_visits', 'comorbidity_count']
    for col in num_cols:
        short = df[df['long_stay'] == 0][col]
        long_stay = df[df['long_stay'] == 1][col]
        t_stat, p_val = stats.ttest_ind(short, long_stay)
        print(f"Feature: {col:18s} | t-stat: {t_stat:8.4f} | p-value: {p_val:.4e}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(df[['age', 'prior_visits', 'comorbidity_count', 'los_days', 'long_stay']].corr(), annot=True, cmap='Blues', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("results/charts/correlation_heatmap.png")
    plt.close()

def build_preprocessor(train_df):
    num_features = ['age', 'prior_visits', 'comorbidity_count']
    cat_features = ['admission_type', 'diagnosis_category']
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])
    X_train_raw = train_df.drop(columns=['los_days', 'long_stay'])
    preprocessor.fit(X_train_raw)
    return preprocessor, num_features, cat_features

def preprocess_df(df, preprocessor):
    X = df.drop(columns=['los_days', 'long_stay'])
    y = df['long_stay'].values
    X_proc = preprocessor.transform(X)
    return X_proc, y

def main():
    print("....FedStay Pipeline Started....")
    setup_environment()
    train_df, test_df = generate_and_split_data()
    run_stats_and_eda(train_df)

    preprocessor, num_features, cat_features = build_preprocessor(train_df)
    X_train, y_train = preprocess_df(train_df, preprocessor)
    X_test, y_test = preprocess_df(test_df, preprocessor)

    print("\n....Training Centralized Random Forest Baseline....")
    central_model = RandomForestClassifier(n_estimators=100, random_state=42)
    central_model.fit(X_train, y_train)
    y_pred_central = central_model.predict(X_test)
    y_prob_central = central_model.predict_proba(X_test)[:, 1]

    acc_central = accuracy_score(y_test, y_pred_central)
    f1_central = f1_score(y_test, y_pred_central)
    auc_central = roc_auc_score(y_test, y_prob_central)

    print("....Training Centralized Logistic Regression....")
    central_lr = LogisticRegression(max_iter=1000, solver='saga', random_state=42)
    central_lr.fit(X_train, y_train)
    y_prob_central_lr = central_lr.predict_proba(X_test)[:, 1]
    auc_central_lr = roc_auc_score(y_test, y_prob_central_lr)

    print("\n....Simulating Federated Learning via Native FedAvg....")
    client_data = []
    for i in range(3):
        site_df = pd.read_csv(f"data/hospital_sites/hospital_site_{i+1}.csv")
        X_s, y_s = preprocess_df(site_df, preprocessor)
        client_data.append((X_s, y_s))

    n_features = X_train.shape[1]
    global_coef = np.zeros((1, n_features))
    global_intercept = np.zeros((1,))

    rounds = 10
    acc_history, auc_history = [], []
    start_time = time.time()

    for r in range(rounds):
        local_coefs, local_intercepts, sizes = [], [], []

        for X_s, y_s in client_data:
            local_model = LogisticRegression(max_iter=500, solver='saga', random_state=42, warm_start=True)
            local_model.fit(X_s, y_s)
            
            local_model.coef_ = global_coef.copy()
            local_model.intercept_ = global_intercept.copy()
            local_model.fit(X_s, y_s)

            local_coefs.append(local_model.coef_)
            local_intercepts.append(local_model.intercept_)
            sizes.append(len(X_s))

        total_samples = sum(sizes)
        global_coef = sum(c * (s / total_samples) for c, s in zip(local_coefs, sizes))
        global_intercept = sum(i * (s / total_samples) for i, s in zip(local_intercepts, sizes))

        eval_model = LogisticRegression()
        eval_model.coef_ = global_coef.copy()
        eval_model.intercept_ = global_intercept.copy()
        eval_model.classes_ = np.array([0, 1])

        probs = eval_model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc_history.append(accuracy_score(y_test, preds))
        auc_history.append(roc_auc_score(y_test, probs))

    fed_time = time.time() - start_time
    y_prob_fed = probs
    y_pred_fed = preds
    acc_fed = accuracy_score(y_test, y_pred_fed)
    f1_fed = f1_score(y_test, y_pred_fed)
    auc_fed = roc_auc_score(y_test, y_prob_fed)

    print("\nResults Summary:")
    print(f"Centralized RF -> Accuracy: {acc_central:.4f} | F1: {f1_central:.4f} | AUC: {auc_central:.4f}")
    print(f"Centralized LR -> AUC: {auc_central_lr:.4f}")
    print(f"Federated LR   -> Accuracy: {acc_fed:.4f} | F1: {f1_fed:.4f} | AUC: {auc_fed:.4f}")
    print(f"Federated Execution Time: {fed_time:.4f}s")

    print("\n....Generating Clean Charts....")
    
    # Chart 1: ROC Comparison
    plt.figure(figsize=(6, 5))
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_central)
    fpr_clr, tpr_clr, _ = roc_curve(y_test, y_prob_central_lr)
    fpr_fl, tpr_fl, _ = roc_curve(y_test, y_prob_fed)
    
    plt.plot(fpr_rf, tpr_rf, label=f'Centralized RF (AUC = {auc_central:.2f})')
    plt.plot(fpr_clr, tpr_clr, label=f'Centralized LR (AUC = {auc_central_lr:.2f})')
    plt.plot(fpr_fl, tpr_fl, label=f'Federated LR (AUC = {auc_fed:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/charts/roc_comparison.png")
    plt.close()

    # Feature Names extraction
    ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = list(num_features) + list(ohe_names)

    # Chart 2: Logistic Regression Coefficients vs Random Forest Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    lr_coef_df = pd.DataFrame({'Coefficient': global_coef.ravel(), 'Feature': feature_names})
    sns.barplot(data=lr_coef_df, x='Coefficient', y='Feature', hue='Feature', palette='mako', legend=False, ax=axes[0])
    axes[0].set_title('Federated LR Model Coefficients')
    
    rf_imp_df = pd.DataFrame({'Importance': central_model.feature_importances_, 'Feature': feature_names})
    sns.barplot(data=rf_imp_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False, ax=axes[1])
    axes[1].set_title('Centralized RF Feature Importances')
    
    plt.tight_layout()
    plt.savefig("results/charts/feature_importance.png")
    plt.close()

    # Chart 3: Federated Convergence Across Rounds Plot
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, rounds + 1), auc_history, marker='o', color='teal', label='Federated Test AUC')
    plt.plot(range(1, rounds + 1), acc_history, marker='s', color='orange', label='Federated Test Accuracy')
    plt.axhline(y=auc_central_lr, color='r', linestyle=':', label='Centralized LR AUC Baseline')
    plt.xlabel('Federated Communication Round')
    plt.ylabel('Score')
    plt.title('FL Convergence Across Rounds')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/charts/fl_convergence.png")
    plt.close()

    print("....FedStay Pipeline Completed Successfully....")

if __name__ == "__main__":
    main()