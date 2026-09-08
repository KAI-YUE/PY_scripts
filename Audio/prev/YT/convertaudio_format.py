import os
from pydub import AudioSegment

input_folder = "/home/kyue/Downloads/recordings/"
output_folder = "/home/kyue/Downloads/tmp/"

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input directory
# for filename in os.listdir(input_folder):
#     if filename.endswith(".ogg"):
#         # Load .ogg file
#         audio = AudioSegment.from_ogg(os.path.join(input_folder, filename))
        
#         # Export as .wav
#         audio.export(os.path.join(output_folder, filename.replace(".ogg", ".wav")), format="wav")

# print("Conversion complete")


for filename in os.listdir(input_folder):
    if filename.endswith(".m4a"):
        # Load .ogg file
        # audio = AudioSegment.from_ogg(os.path.join(input_folder, filename))
        audio = AudioSegment.from_file(os.path.join(input_folder, filename), format='m4a')
        
        # Export as .wav
        audio.export(os.path.join(output_folder, filename.replace(".m4a", ".wav")), format="wav")

print("Conversion complete")