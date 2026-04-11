from moviepy import VideoFileClip

def convert_to_wav(input_file, output_file):
    # Load the video clip
    video = VideoFileClip(input_file)

    # Extract the audio from the video clip
    audio = video.audio

    # Write the audio to a WAV file
    # audio.write_audiofile(output_file, codec='pcm_s16le')

    # Write the audio to an MP3 file
    audio.write_audiofile(output_file, codec='libmp3lame')

    print("Conversion completed successfully!")

file_name = "videoplayback"

# Usage example
input_file = r"/home/kyue/Downloads/{:s}.mp4".format(file_name)

# MP3 file
output_file = r"/home/kyue/Downloads/{:s}.mp3".format(file_name)
convert_to_wav(input_file, output_file)
