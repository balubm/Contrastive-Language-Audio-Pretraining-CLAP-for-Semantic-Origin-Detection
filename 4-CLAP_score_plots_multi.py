import pandas as pd
import matplotlib.pyplot as plt
import os

input_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
output_folder = input_folder
os.makedirs(output_folder, exist_ok=True)

# Find the first CSV file in the folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV files found in the specified input folder.")

# Use the first CSV file found
csv_path = os.path.join(input_folder, csv_files[0])
df = pd.read_csv(csv_path)

score_columns = [
    "score_AI_generated_synthetic_music",
    "score_Human_performed_live_music",
    "score_Music_with_audio_glitches_clicks_and_pops",
    "score_Music_with_smooth_production",
    "score_Harmonic_content_dominant",
    "score_Noise_dominant",
    "score_Tonal_harmonic_music",
    "score_Atonal_noise_music"
]
existing_score_columns = [col for col in score_columns if col in df.columns]
if not existing_score_columns:
    raise ValueError("None of the score columns were found in the CSV.")

# Sanity check source column
if 'source' not in df.columns:
    raise ValueError("No 'source' column found (required for Suno/Human grouping).")

# Draw boxplots
plt.figure(figsize=(18, 10))
data_to_plot = []

xticklabels = []
for col in existing_score_columns:
    for source in ['Suno', 'Human']:
        if source in df['source'].unique():
            group_data = df.loc[df['source'] == source, col].dropna()
            data_to_plot.append(group_data.values)
            xticklabels.append(f"{col}\n({source})")

plt.boxplot(data_to_plot, patch_artist=True)
plt.xticks(range(1, len(xticklabels) + 1), xticklabels, rotation=75, ha='right')
plt.xlabel("Score Columns by Source")
plt.ylabel("Score Value")
plt.title("Score Distributions by Feature and Source (Suno vs Human)")
plt.tight_layout()

# Save and show
plot_path = os.path.join(output_folder, "audio_scores_boxplot_suno_human.png")
plt.savefig(plot_path, dpi=200)
plt.show()
