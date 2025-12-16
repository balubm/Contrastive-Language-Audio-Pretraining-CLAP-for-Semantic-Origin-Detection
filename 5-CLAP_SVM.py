import os
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

# Folder containing your CSV files
input_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Load and merge CSVs on filename and source
dfs = []
for csv_file in csv_files:
    print(f"📥 Loading: {csv_file}")
    path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(path)
    dfs.append(df[["filename", "source"] + [c for c in df.columns if c.startswith("score_")]])

df_merged = reduce(lambda left, right: pd.merge(left, right, on=["filename", "source"]), dfs)
df_merged.to_csv("df_merged.csv", index=False)

# Prepare features and labels
feature_cols = [c for c in df_merged.columns if c.startswith("score_")]
print("Feature columns:", feature_cols)
X = df_merged[feature_cols]
y = df_merged["source"]

# Check minimum samples per class
if min(y.value_counts()) < 2:
    print("ERROR: Each class must have at least 2 samples for stratification. Found:", y.value_counts())
else:
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Define SVM and parameter grid for GridSearch
    svm = SVC(kernel='rbf', random_state=42)
    param_grid = {'C': [100], 'gamma': [0.01]}
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_svm = grid_search.best_estimator_
    print("Best C parameter:", grid_search.best_params_['C'])
    print("Best gamma parameter:", grid_search.best_params_['gamma'])

    # Predict and evaluate on test set
    y_pred = best_svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose().loc[['Suno','Human']][['precision','recall','f1-score','support']]
    cr_df = cr_df.rename_axis('Source').reset_index()
    cr_df = cr_df.round(2)

    # Insert accuracy row at top or bottom as needed
    acc_row = pd.DataFrame({'Source':['Overall Accuracy'],'precision':[acc],
                            'recall':[''], 'f1-score':[''], 'support':['']})
    cr_df = pd.concat([acc_row, cr_df], ignore_index=True)

    # Save and display
    cr_df.to_csv("svm_gridsearch_results_full.csv", index=False)
    print("Grid search results with support saved to svm_gridsearch_results_full.csv")

    print("\nResults Table (with support):")
    print(cr_df)

# Extract TP, TN, FP, FN from confusion matrix for binary classification
# Assuming 'Suno' and 'Human' are your classes and appear in that order in y_test.unique()
# Confusion matrix layout for binary classification:
# cm = [[TN, FP],
#       [FN, TP]]

# Get class index order to map confusion matrix correctly
class_labels = list(best_svm.classes_)
tn = cm[class_labels.index('Human'), class_labels.index('Human')]
tp = cm[class_labels.index('Suno'), class_labels.index('Suno')]
fp = cm[class_labels.index('Human'), class_labels.index('Suno')]
fn = cm[class_labels.index('Suno'), class_labels.index('Human')]

conf_matrix_df = pd.DataFrame([
    {'Metric': 'True Positive',  'Count': tp},
    {'Metric': 'True Negative',  'Count': tn},
    {'Metric': 'False Positive', 'Count': fp},
    {'Metric': 'False Negative', 'Count': fn}
])


# Save confusion matrix counts as CSV
conf_matrix_df.to_csv("svm_confusion_matrix_counts.csv", index=False)
print("\nConfusion matrix counts saved to svm_confusion_matrix_counts.csv")

print("\nConfusion Matrix Counts:")
print(conf_matrix_df)
