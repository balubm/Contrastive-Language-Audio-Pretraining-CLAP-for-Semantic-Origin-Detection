import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG ====
input_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
output_folder = input_folder
os.makedirs(output_folder, exist_ok=True)

csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
sns.set(style="whitegrid", font_scale=1.15)

for csv_file in csv_files:
    csv_path = os.path.join(input_folder, csv_file)
    print(f"📊 Generating scatter plots with thin plus markers and %: {csv_path}")

    df = pd.read_csv(csv_path)
    base_name = os.path.splitext(csv_file)[0]
    parts = base_name.split("_vs_")
    if len(parts) != 2:
        print(f"⚠️ Skipping file {csv_file} (invalid naming format)")
        continue

    score_col_1 = f"score_{parts[0]}"
    score_col_2 = f"score_{parts[1]}"
    label_col_1 = parts[0].replace("_", " ").title()
    label_col_2 = parts[1].replace("_", " ").title()

    pred_labels = sorted(df["predicted_label"].unique())
    palette = sns.color_palette("Set2", len(pred_labels))
    pred_to_color = dict(zip(pred_labels, palette))

    suno_df = df[df["source"] == "Suno"]
    human_df = df[df["source"] == "Human"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    min_val = min(df[score_col_1].min(), df[score_col_2].min())
    max_val = max(df[score_col_1].max(), df[score_col_2].max())

    # Suno Source - Thin "+" markers
    for pred_label in pred_labels:
        mask = suno_df["predicted_label"] == pred_label
        axes[0].scatter(
            suno_df.loc[mask, score_col_1],
            suno_df.loc[mask, score_col_2],
            color=pred_to_color[pred_label],
            label=pred_label,
            alpha=0.7,
            s=20,             # SMALL size
            linewidths=0.7,   # THIN lines
            marker='+'
        )
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    axes[0].set_title("Suno Source")
    axes[0].set_xlabel(f"{label_col_1} Score")
    axes[0].set_ylabel(f"{label_col_2} Score")

    # Percentage annotation for Suno (inside plot, top-left)
    total_suno = len(suno_df)
    annot_y = max_val - (max_val - min_val) * 0.08
    for i, pred_label in enumerate(pred_labels):
        count = (suno_df["predicted_label"] == pred_label).sum()
        perc = 100 * count / total_suno if total_suno > 0 else 0
        axes[0].text(
            min_val + (max_val - min_val) * 0.03,
            annot_y - i * (max_val - min_val) * 0.08,
            f"{pred_label}: {perc:.1f}%",
            color=pred_to_color[pred_label],
            fontsize=12,
            fontweight='bold',
            va='top'
        )

    # Human Source - Thin "+" markers
    for pred_label in pred_labels:
        mask = human_df["predicted_label"] == pred_label
        axes[1].scatter(
            human_df.loc[mask, score_col_1],
            human_df.loc[mask, score_col_2],
            color=pred_to_color[pred_label],
            label=pred_label,
            alpha=0.7,
            s=20,             # SMALL size
            linewidths=0.7,   # THIN lines
            marker='+'
        )
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    axes[1].set_title("Human Source")
    axes[1].set_xlabel(f"{label_col_1} Score")
    axes[1].set_ylabel(f"{label_col_2} Score")

    # Percentage annotation for Human (inside plot, top-left)
    total_human = len(human_df)
    for i, pred_label in enumerate(pred_labels):
        count = (human_df["predicted_label"] == pred_label).sum()
        perc = 100 * count / total_human if total_human > 0 else 0
        axes[1].text(
            min_val + (max_val - min_val) * 0.03,
            annot_y - i * (max_val - min_val) * 0.08,
            f"{pred_label}: {perc:.1f}%",
            color=pred_to_color[pred_label],
            fontsize=12,
            fontweight='bold',
            va='top'
        )

    # Shared legend for colors
    handles = [plt.Line2D([0], [0], marker='+', color='w', label=lbl,
                          markerfacecolor=pred_to_color[lbl], markeredgecolor=pred_to_color[lbl],
                          markersize=10, linewidth=0.7)
               for lbl in pred_labels]
    fig.legend(handles, pred_labels, loc="upper center", ncol=len(pred_labels), fontsize=12, title="Predicted Label")

    #plt.suptitle(f"Scatter Plot by Source: {label_col_1} vs {label_col_2}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    file_out = os.path.join(output_folder, f"{base_name}_scatter_by_source_plus_thin.png")
    plt.savefig(file_out, dpi=150)
    plt.close()

    print(f"✅ Saved scatter plot (thin plus markers): {file_out}")
    print("=" * 30)

print("\n🎨 All scatter plots with thin plus markers and internal percentage annotations generated successfully.")
