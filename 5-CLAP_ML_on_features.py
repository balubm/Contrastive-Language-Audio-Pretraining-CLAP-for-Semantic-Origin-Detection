import os
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

input_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

dfs = []
for csv_file in csv_files:
    print(f"📥 Loading: {csv_file}")
    path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(path)
    
    score_cols = [c for c in df.columns if c.startswith("score_")]
    # for col in score_cols:
    #     df.rename(columns={col: f"{col}_{csv_file.replace('.csv','')}"}, inplace=True)
    
    dfs.append(df[["filename", "source"] + [c for c in df.columns if c.startswith("score_")]])

df_merged = reduce(lambda left, right: pd.merge(left, right, on=["filename", "source"]), dfs)
df_merged.to_csv("df_merged.csv", index=False)
feature_cols = [c for c in df_merged.columns if c.startswith("score_")]
print(feature_cols)
X = df_merged[feature_cols]
y = df_merged["source"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

models = {
    "SVM (SVC)": SVC(kernel='linear', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # Flatten metrics for CSV output
    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Human_Precision": cr["Human"]["precision"],
        "Human_Recall": cr["Human"]["recall"],
        "Human_F1": cr["Human"]["f1-score"],
        "Suno_Precision": cr["Suno"]["precision"],
        "Suno_Recall": cr["Suno"]["recall"],
        "Suno_F1": cr["Suno"]["f1-score"]
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df = results_df.round(2) 
results_df.to_csv("model_evaluation_results.csv", index=False)
print("Results saved to model_evaluation_results.csv")
