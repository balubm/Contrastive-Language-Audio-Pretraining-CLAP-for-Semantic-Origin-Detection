import os
import pandas as pd

input_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Dynamic ground truth mapping function
def true_label(source, label1, label2):
    if source == "Suno":
        return label1
    elif source == "Human":
        return label2

for csv_file in csv_files:
    path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(path)

    # Dynamically find class labels from column names
    class_cols = [col for col in df.columns if col.startswith("score_")]
    # Remove the 'score_' prefix for labels
    label1 = class_cols[0].replace("score_", "").replace("_", " ")
    label2 = class_cols[1].replace("score_", "").replace("_", " ")

    # Assign true label column
    df["true_label"] = df["source"].apply(true_label, args=(label1, label2))
    df["correct"] = df["predicted_label"] == df["true_label"]
    accuracy = df["correct"].mean()
    print(f"{csv_file}: {accuracy:.2%}")

