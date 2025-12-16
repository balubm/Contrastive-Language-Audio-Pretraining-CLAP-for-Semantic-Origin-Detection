import os
import csv
import torch
from tqdm import tqdm
from msclap import CLAP
import torchaudio
import pandas as pd
import tempfile


# 1. Define class label pairs (customize as needed)
class_label_pairs = [
    ["ai-generated music", "human-performed music"],
    ["digital", "natural, live"],
    ["music with audio glitches", "music with smooth production"],
    ["synthetic music track", "music created by a musician"],
]


# 2. Paths to folders
suno_folder = "/Volumes/QL SSD1/ai_human_music_data/Suno21k/archive/audio"
human_folder = "/Volumes/QL SSD1/ai_human_music_data/humanFMA/archive/fma_medium/fma_medium"
#suno_folder = "/Volumes/QL SSD1/ai_human_music_data/suno_tracks"
#human_folder = "/Volumes/QL SSD1/ai_human_music_data/human_tracks"
output_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/CLAP_detection/plots"

os.makedirs(output_folder, exist_ok=True)


# 3. Collect valid .wav/.mp3 files recursively (handling subfolders)
def collect_files(folder, source_label):
    valid_files = []
    count=0
    for root, _, files in os.walk(folder):
        for f in files:
            if not (f.endswith(".wav") or f.endswith(".mp3")):
                continue
            path = os.path.join(root, f)
            try:
                waveform, sample_rate = torchaudio.load(path)
                duration_sec = waveform.shape[1] / sample_rate
                if duration_sec >= 1:
                    valid_files.append((path, source_label, sample_rate))
                    count+=1
                    if count % 500 == 0:
                        print(f"✅ Collected {count} valid files so far...")
            except Exception as e:
                print(f"⚠️ Skipping {f} (unreadable): {e}")
   
    return valid_files


suno_files = collect_files(suno_folder, "Suno")
human_files = collect_files(human_folder, "Human")
all_files = suno_files + human_files

file_paths    = [f[0] for f in all_files]
sources       = [f[1] for f in all_files]
sample_rates  = [f[2] for f in all_files]

print(f"✅ Found {len(file_paths)} valid audio files to analyze.")


# --- Extract last 30s excerpt function ---
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


# Modified function: process one file at a time, delete temp file immediately
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


# Get audio embeddings
audio_embeddings = get_audio_embeds_single(clap_model, file_paths, sample_rates)


# 5. Main Loop: Compute & Save CSVs for each class label pair
for class_labels in class_label_pairs:
    print(f"\nProcessing: {class_labels}")
    text_embeddings = clap_model.get_text_embeddings(class_labels)
    similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

    df = pd.DataFrame({
        "filename": [os.path.basename(p) for p in file_paths],
        "source": sources,
        f"score_{class_labels[0].replace(' ', '_')}": similarities[:, 0].detach().numpy(),
        f"score_{class_labels[1].replace(' ', '_')}": similarities[:, 1].detach().numpy(),
        "predicted_label": [class_labels[i] for i in similarities.argmax(axis=1)]
    })

    csv_name = f"{class_labels[0].replace(' ', '_')}_vs_{class_labels[1].replace(' ', '_')}.csv"
    csv_path = os.path.join(output_folder, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV: {csv_path}")

print("\n✅ CSV generation finished. No plots were created.")
