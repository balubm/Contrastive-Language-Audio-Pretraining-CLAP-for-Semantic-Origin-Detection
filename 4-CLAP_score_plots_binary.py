import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==== CONFIG ====
input_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
output_folder = input_folder
os.makedirs(output_folder, exist_ok=True)

csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
sns.set(style="whitegrid", font_scale=1.15)

for csv_file in csv_files:
    csv_path = os.path.join(input_folder, csv_file)
    print(f"📊 Plotting from: {csv_path}")

    df = pd.read_csv(csv_path)
    base_name = os.path.splitext(csv_file)[0]
    parts = base_name.split("_vs_")
    if len(parts) != 2:
        print(f"⚠️ Skipping file {csv_file} (invalid naming format)")
        continue

    class1 = parts[0].replace("_", " ")
    class2 = parts[1].replace("_", " ")

    score_col_1 = f"score_{parts[0]}"
    score_col_2 = f"score_{parts[1]}"

    def keep_highest(df_sub):
        cmp = df_sub[[score_col_1, score_col_2]].values
        max_idx = np.argmax(cmp, axis=1)
        mask1 = max_idx != 0
        mask2 = max_idx != 1
        out_df = df_sub.copy()
        out_df.loc[mask1, score_col_1] = np.nan
        out_df.loc[mask2, score_col_2] = np.nan
        return out_df

    suno_df = df[df["source"].str.lower() == "suno"]
    human_df = df[df["source"].str.lower() == "human"]

    suno_high = keep_highest(suno_df)
    human_high = keep_highest(human_df)

    suno_melted = suno_high.melt(
        id_vars=["filename", "source"],
        value_vars=[score_col_1, score_col_2],
        var_name="Class",
        value_name="Score"
    ).dropna(subset=["Score"])

    human_melted = human_high.melt(
        id_vars=["filename", "source"],
        value_vars=[score_col_1, score_col_2],
        var_name="Class",
        value_name="Score"
    ).dropna(subset=["Score"])

    suno_melted["Class"] = suno_melted["Class"].str.replace("score_", "").str.replace("_", " ")
    human_melted["Class"] = human_melted["Class"].str.replace("score_", "").str.replace("_", " ")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sns.boxplot(data=suno_melted, x="Class", y="Score", showfliers=False, ax=axes[0])
    sns.stripplot(data=suno_melted, x="Class", y="Score", color='black', alpha=0.5, ax=axes[0])
    axes[0].set_title("Suno songs")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Highest CLAP Similarity Score")
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].set_ylim(-5, 20)  # Set y-axis range here

    # Annotate Suno box plot with data point counts
    for i, label in enumerate(suno_melted["Class"].unique()):
        count = suno_melted[suno_melted["Class"] == label].shape[0]
        axes[0].text(i, axes[0].get_ylim()[1] - 1, f'n={count}', ha='center', va='top', fontsize=10, color='darkblue')

    sns.boxplot(data=human_melted, x="Class", y="Score", showfliers=False, ax=axes[1])
    sns.stripplot(data=human_melted, x="Class", y="Score", color='black', alpha=0.5, ax=axes[1])
    axes[1].set_title("Human songs")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Highest CLAP Similarity Score")
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].set_ylim(-5, 20)  # Set y-axis range here

    # Annotate Human box plot with data point counts
    for i, label in enumerate(human_melted["Class"].unique()):
        count = human_melted[human_melted["Class"] == label].shape[0]
        axes[1].text(i, axes[1].get_ylim()[1] - 1, f'n={count}', ha='center', va='top', fontsize=10, color='darkblue')

    plt.suptitle(f"CLAP Class Comparison (Only Highest): {class1.title()} vs {class2.title()}", fontsize=12, y=1.02)
    plt.tight_layout()

    fig_name = f"{base_name}_sourcewise_only_highest_plot.png"
    fig_path = os.path.join(output_folder, fig_name)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved plot: {fig_path}")
    print("=" * 30)

print("\n🎨 All highest-only source-wise comparison plots generated successfully.")
