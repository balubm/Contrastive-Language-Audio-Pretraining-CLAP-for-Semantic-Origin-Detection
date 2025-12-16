import os
import torchaudio
import pandas as pd

# Define your folders here
suno_folder = "/Volumes/QL SSD1/ai_human_music_data/Suno21k/archive/audio"
human_folder = "/Volumes/QL SSD1/ai_human_music_data/humanFMA/archive/fma_medium/fma_medium"

def collect_files(folder, source_label):
    valid_files = []
    count = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if not (f.endswith(".wav") or f.endswith(".mp3")):
                continue
            path = os.path.join(root, f)
            try:
                waveform, sample_rate = torchaudio.load(path)
                duration_sec = waveform.shape[1] / sample_rate
                if duration_sec >= 1:
                    valid_files.append({
                        "filepath": path,
                        "source": source_label,
                        "sample_rate": sample_rate
                    })
                    count += 1
                    if count % 500 == 0:
                        print(f"✅ Collected {count} valid files so far from {source_label}...")
            except Exception as e:
                print(f"⚠️ Skipping {f} in {source_label} (unreadable): {e}")

    return valid_files

def main():
    suno_files = collect_files(suno_folder, "Suno")
    human_files = collect_files(human_folder, "Human")
    all_files = suno_files + human_files

    df = pd.DataFrame(all_files)
    output_csv = "audio_files_metadata.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved combined metadata CSV: {output_csv}")

if __name__ == "__main__":
    main()
