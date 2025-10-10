import os
import soundfile as sf
from datasets import load_dataset, Audio

# --- Configuration ---
# 1. Make sure this path points to your output folder.
output_folder = r"C:\Users\ShwethaSharan T\Desktop\MY PROJECTS\Blue Fish\Blue-Fish\marine_input"

# 2. Name of the Hugging Face dataset
repo_id = "ardavey/marine_ocean_mammal_sound"
# --- End of Configuration ---

print("➡️ Loading dataset metadata (using simple method)...")
# Load the dataset but tell it NOT to decode the audio automatically
# This is the key change that avoids the error.
dataset = load_dataset(repo_id, trust_remote_code=True).cast_column("audio", Audio(decode=False))

print("➡️ Starting export process...")
file_counters = {}

for item in dataset['train']:
    try:
        # Get the path to the original audio file in the cache
        original_audio_path = item['audio']['path']
        species_label = item['species']

        # Use soundfile to directly read the original audio file
        audio_array, sampling_rate = sf.read(original_audio_path)

        # --- The rest of the saving logic is the same ---
        folder_name = species_label.replace(" ", "_")
        species_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(species_folder_path, exist_ok=True)

        count = file_counters.get(folder_name, 0) + 1
        file_counters[folder_name] = count

        output_filename = f"{folder_name}_{count}.wav"
        output_filepath = os.path.join(species_folder_path, output_filename)

        # Write the audio data as a new WAV file
        sf.write(output_filepath, audio_array, sampling_rate)
        print(f"Successfully saved: {output_filename}")

    except Exception as e:
        print(f"Could not process a file. Error: {e}")


print(f"\n✅ Export complete! Audio files are saved in: {output_folder}")