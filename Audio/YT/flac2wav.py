import os
from pydub import AudioSegment

input_folder = "/home/kyue/Audio/jaychou/tmp/"
output_folder = "/home/kyue/Audio/jaychou/"

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input directory
for filename in os.listdir(input_folder):
    if filename.endswith(".flac"):
        # Load .flac file
        audio = AudioSegment.from_file(os.path.join(input_folder, filename), format="flac")
        
        # Export as .wav
        audio.export(os.path.join(output_folder, filename.replace(".flac", ".mp3")), format="mp3")

print("Conversion complete")
