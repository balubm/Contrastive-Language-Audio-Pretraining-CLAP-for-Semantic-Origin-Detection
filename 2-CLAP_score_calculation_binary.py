import os
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
    ["harmonic content", "noise"],
    ["tonal", "atonal"],]

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



# 7. Main Loop: Compute & Save CSVs for each class label pair
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