import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss
)

from sklearn.calibration import calibration_curve

from pytorch_tabnet.tab_model import TabNetClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# -----------------------------
# Step 1: Load dataset
# -----------------------------

file_path = "mycotoxin_dataset.xlsx"

df = pd.read_excel(file_path)


X = df[
    [
        'Aflatoxin_ng_mL',
        'OchratoxinA_ng_mL',
        'Zearalenone_ng_mL',
        'Fumonisin_Biomarker_Ratio'
    ]
].values


y = df['Exposure_Label'].values



# Encode labels

le = LabelEncoder()

y = le.fit_transform(y)



# -----------------------------
# Step 2: Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




# -----------------------------
# Step 3: TabNet Model
# -----------------------------

tabnet_model = TabNetClassifier(

    n_d=8,
    n_a=8,

    n_steps=3,

    gamma=1.5,

    n_independent=2,
    n_shared=2,

    mask_type='entmax',

    seed=42
)



# -----------------------------
# Step 4: Train model
# -----------------------------

tabnet_model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_test,y_test)
    ],

    eval_metric=['accuracy'],

    max_epochs=100,

    patience=20,

    batch_size=16,

    virtual_batch_size=8,

    num_workers=0
)




# -----------------------------
# Step 5: Predictions
# -----------------------------

y_test_pred = tabnet_model.predict(
    X_test
).squeeze()


y_test_proba = tabnet_model.predict_proba(
    X_test
)[:,1]




# Save predictions

df['Predicted_Label'] = le.inverse_transform(
    tabnet_model.predict(X).squeeze()
)


df.to_excel(
    "tabnet_mycotoxin_predictions.xlsx",
    index=False
)





# -----------------------------
# Step 6: Accuracy
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_test_pred
)


print(
    "TabNet Accuracy:",
    accuracy
)




# -----------------------------
# Step 7: Classification Report
# -----------------------------

report = classification_report(

    y_test,

    y_test_pred,

    target_names=le.classes_,

    output_dict=True
)



report_df = pd.DataFrame(report).transpose()



report_df = report_df.drop(
    columns=['support']
)



print("\nClassification Report")

print(report_df)






# -----------------------------
# Step 8: Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_test_pred
)


plt.figure(figsize=(6,5))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=le.classes_,

    yticklabels=le.classes_

)


plt.title(
    "Confusion Matrix - TabNet"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.show()





# -----------------------------
# Step 9: ROC Curve + AUC
# -----------------------------


fpr,tpr,_ = roc_curve(
    y_test,
    y_test_proba
)


roc_auc = auc(
    fpr,
    tpr
)



plt.figure(figsize=(6,5))


plt.plot(

    fpr,

    tpr,

    label=f"AUC={roc_auc:.3f}"

)



plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)



plt.title(
    "ROC Curve - TabNet"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.legend()

plt.show()





# -----------------------------
# Step 10: Precision Recall Curve
# -----------------------------


precision, recall, _ = precision_recall_curve(

    y_test,

    y_test_proba

)



plt.figure(figsize=(6,5))


plt.plot(
    recall,
    precision
)



plt.title(
    "Precision Recall Curve - TabNet"
)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.show()





# -----------------------------
# Step 11: PR-AUC
# -----------------------------


pr_auc = average_precision_score(

    y_test,

    y_test_proba

)


print(
    "PR-AUC:",
    pr_auc
)





# -----------------------------
# Step 12: Calibration Metric
# Brier Score
# -----------------------------


brier = brier_score_loss(

    y_test,

    y_test_proba

)


print(
    "Brier Score:",
    brier
)






# -----------------------------
# Step 13: Calibration Curve
# -----------------------------


prob_true, prob_pred = calibration_curve(

    y_test,

    y_test_proba,

    n_bins=10

)



plt.figure(figsize=(6,5))


plt.plot(

    prob_pred,

    prob_true,

    marker='o',

    label="TabNet Calibration"

)



plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    label="Perfect Calibration"

)



plt.title(
    "Calibration Curve - TabNet"
)


plt.xlabel(
    "Predicted Probability"
)


plt.ylabel(
    "Observed Probability"
)


plt.legend()

plt.show()





# -----------------------------
# Step 14: Feature Importance
# -----------------------------


features = [

    'Aflatoxin',

    'OchratoxinA',

    'Zearalenone',

    'Fumonisin_Ratio'

]


importance = tabnet_model.feature_importances_



plt.figure(figsize=(7,5))


sns.barplot(

    x=importance,

    y=features

)


plt.title(
    "Feature Importance - TabNet"
)


plt.xlabel(
    "Importance Score"
)


plt.ylabel(
    "Features"
)


plt.show()





# -----------------------------
# Step 15: TabNet Explainability
# -----------------------------


explain_matrix, masks = tabnet_model.explain(
    X_test
)


plt.figure(figsize=(8,5))


sns.heatmap(
    explain_matrix,
    cmap="viridis"
)


plt.title(
    "TabNet Feature Attribution Heatmap"
)


plt.xlabel(
    "Features"
)


plt.ylabel(
    "Samples"
)


plt.show()





# -----------------------------
# Final Evaluation Summary
# -----------------------------


print("\n===== Final TabNet Evaluation =====")

print("Accuracy :", accuracy)

print("ROC-AUC  :", roc_auc)

print("PR-AUC   :", pr_auc)

print("Brier Score :", brier)