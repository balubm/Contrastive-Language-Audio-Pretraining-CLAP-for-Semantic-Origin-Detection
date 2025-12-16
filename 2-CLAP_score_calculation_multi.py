import os
import torch
from tqdm import tqdm
from msclap import CLAP
import torchaudio
import pandas as pd
import tempfile
from datetime import datetime
# Print time before running
print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 1. Define class label pairs (customize as needed)
class_labels_multis = [
    "AI_generated_synthetic_music",
    "Human_performed_live_music",
    "Music_with_audio_glitches_clicks_and_pops",
    "Music_with_smooth_production",
    "Harmonic_content_dominant",
    "Noise_dominant",
    "Tonal_harmonic_music",
    "Atonal_noise_music"
]

# 2. Load saved metadata CSV
metadata_csv_path = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/Out/audio_files_metadata.csv"  # from your extract script
output_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"
os.makedirs(output_folder, exist_ok=True)

df_metadata = pd.read_csv(metadata_csv_path)
file_paths = df_metadata["filepath"].tolist()
sources = df_metadata["source"].tolist()
sample_rates = df_metadata["sample_rate"].tolist()

print(f"✅ Loaded metadata for {len(file_paths)} audio files.")


# 3. Extract last 30s excerpt function (unchanged)
def extract_last_30s(input_path, sample_rate):
    waveform, sr = torchaudio.load(input_path)
    total_length = waveform.shape[1]
    last_30s_frames = 30 * sr
    if total_length > last_30s_frames:
        excerpt = waveform[:, -last_30s_frames:]
    else:
        excerpt = waveform
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)
    torchaudio.save(temp_path, excerpt, sr)
    return temp_path


# 4. Initialize CLAP model
clap_model = CLAP(version='2023', use_cuda=False)


# 5. Audio embedding extraction (same, processes files individually and cleans up temp files)
def get_audio_embeds_single(model, files, sample_rates):
    all_embeds = []
    for i, path in enumerate(tqdm(files, desc="🔊 Processing audio files")):
        sr = sample_rates[i]
        temp_excerpt = extract_last_30s(path, sr)
        try:
            embed = model.get_audio_embeddings([temp_excerpt])
            all_embeds.append(embed)
        finally:
            try:
                os.remove(temp_excerpt)
            except Exception as e:
                print(f"⚠️ Failed to delete temp file {temp_excerpt}: {e}") 
    return torch.cat(all_embeds, dim=0)


# 6. Get audio embeddings
audio_embeddings = get_audio_embeds_single(clap_model, file_paths, sample_rates)


# 7. Main Loop: Compute & Save CSV for multiclass classification

print(f"\nProcessing multiclass classification with {len(class_labels_multis)} classes")
text_embeddings = clap_model.get_text_embeddings(class_labels_multis)
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

df = pd.DataFrame({
    "filename": [os.path.basename(p) for p in file_paths],
    "source": sources,
})

# Add similarity scores per class as separate columns
for i, label in enumerate(class_labels_multis):
    col_name = f"score_{label.replace(' ', '_')}"
    df[col_name] = similarities[:, i].detach().numpy()

csv_path = os.path.join(output_folder, "multiclass_scores_and_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"💾 Saved multiclass CSV: {csv_path}")
print("\n✅ CSV generation for multiclass classification finished.")

# Print time after running
print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))