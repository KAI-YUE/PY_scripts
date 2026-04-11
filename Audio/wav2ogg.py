import os
from pydub import AudioSegment

input_folder = "/mnt/ssd/HMeshi/_6_Lua/HM/resources/test/"
output_folder = "/mnt/ssd/HMeshi/_6_Lua/HM/resources/test/output/"

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input directory
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        # Load .wav file
        audio = AudioSegment.from_file(os.path.join(input_folder, filename), format="wav")
        
        # Export
        audio.export(os.path.join(output_folder, filename.replace(".wav", ".ogg")), format="ogg")

print("Conversion complete")
