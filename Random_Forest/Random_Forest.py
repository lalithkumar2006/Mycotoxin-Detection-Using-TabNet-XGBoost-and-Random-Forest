# ==========================================
# Random Forest for Mycotoxin Detection
# ==========================================


# -----------------------------
# Step 1: Import Libraries
# -----------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestClassifier


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





# -----------------------------
# Step 2: Load Dataset
# -----------------------------

file_path = "mycotoxin_dataset.xlsx"

df = pd.read_excel(file_path)



# Features

X = df[
    [
        'Aflatoxin_ng_mL',
        'OchratoxinA_ng_mL',
        'Zearalenone_ng_mL',
        'Fumonisin_Biomarker_Ratio'
    ]
].values



# Target

y = df['Exposure_Label'].values




# Encode Labels

le = LabelEncoder()

y = le.fit_transform(y)





# -----------------------------
# Step 3: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)





# -----------------------------
# Step 4: Random Forest Model
# -----------------------------

rf_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    random_state=42,

    class_weight="balanced"

)




# Train

rf_model.fit(

    X_train,

    y_train

)







# -----------------------------
# Step 5: Predictions
# -----------------------------

y_test_pred = rf_model.predict(

    X_test

)


y_test_proba = rf_model.predict_proba(

    X_test

)[:,1]





# Save predictions

df['Predicted_Label_RF'] = le.inverse_transform(

    rf_model.predict(X)

)



df.to_excel(

    "random_forest_mycotoxin_predictions.xlsx",

    index=False

)








# -----------------------------
# Step 6: Accuracy
# -----------------------------


accuracy = accuracy_score(

    y_test,

    y_test_pred

)



print("\nRandom Forest Accuracy:",
      accuracy)







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



print("\nClassification Report:")

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
    "Confusion Matrix - Random Forest"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.show()








# -----------------------------
# Step 9: ROC Curve
# -----------------------------


fpr, tpr, _ = roc_curve(

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

    label=f"AUC = {roc_auc:.3f}"

)



plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)



plt.title(
    "ROC Curve - Random Forest"
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
    "Precision-Recall Curve - Random Forest"
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

    label="Random Forest Calibration"

)



plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    label="Perfect Calibration"

)



plt.title(
    "Calibration Curve - Random Forest"
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



importance = rf_model.feature_importances_






plt.figure(figsize=(7,5))



sns.barplot(

    x=importance,

    y=features

)



plt.title(
    "Feature Importance - Random Forest"
)



plt.xlabel(
    "Importance Score"
)



plt.ylabel(
    "Features"
)



plt.show()







# -----------------------------
# Step 15: Final Results
# -----------------------------


print("\nFinal Random Forest Results")

print("--------------------------")


print("Accuracy :", accuracy)

print("ROC-AUC  :", roc_auc)

print("PR-AUC   :", pr_auc)

print("Brier Score :", brier)