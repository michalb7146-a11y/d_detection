import os
from pydub import AudioSegment

# Define folder paths
input_folder = "/Users/deviceone/Downloads/data/tut/tut_audio"
output_folder = "/Users/deviceone/Downloads/data/tut/new_balanced_2s_dataset_tut/background"

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Segment length in milliseconds (2 seconds = 2000 ms)
segment_length_ms = 2000 

# Scan for files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing file: {filename}")
        
        # Load the audio file
        audio = AudioSegment.from_wav(file_path)
        
        # Get total duration
        duration_ms = len(audio)
        
        # Split the file into 2-second chunks
        start = 0
        counter = 1
        while start < duration_ms:
            end = start + segment_length_ms
            
            # Slice the audio
            chunk = audio[start:end]
            
            # Generate new filename (e.g., original_1.wav)
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}_{counter}.wav"
            output_path = os.path.join(output_folder, output_filename)
            
            # Export the chunk
            chunk.export(output_path, format="wav")
            
            start = end
            counter += 1

print("Done! All sliced files are saved in the 'cut_files' folder.")