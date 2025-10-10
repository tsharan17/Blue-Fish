import os
import soundfile as sf
from datasets import load_dataset

# --- Configuration ---
# 1. CHANGE THIS PATH to the folder you created in Step 1.
output_folder = r"C:\Users\ShwethaSharan T\Desktop\MY PROJECTS\Blue Fish\Blue-Fish\marine_input"

# 2. Name of the Hugging Face dataset
repo_id = "ardavey/marine_ocean_mammal_sound"
# --- End of Configuration ---

print("➡️ Loading Hugging Face dataset...")
dataset = load_dataset(repo_id)

print("➡️ Starting export process...")
file_counters = {}

# Make sure the lines below this start with 4 spaces
for item in dataset['train']:
    audio_data = item['audio']
    species_label = item['species']

    folder_name = species_label.replace(" ", "_")
    species_folder_path = os.path.join(output_folder, folder_name)

    os.makedirs(species_folder_path, exist_ok=True)

    count = file_counters.get(folder_name, 0) + 1
    file_counters[folder_name] = count

    output_filename = f"{folder_name}_{count}.wav"
    output_filepath = os.path.join(species_folder_path, output_filename)

    sf.write(
        output_filepath,
        audio_data['array'],
        audio_data['sampling_rate']
    )

print(f"\n✅ Export complete! Audio files are saved in: {output_folder}")