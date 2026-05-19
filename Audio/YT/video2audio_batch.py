from moviepy import VideoFileClip

def convert_to_wav(input_file, output_file):
    # Load the video clip
    video = VideoFileClip(input_file)

    # Extract the audio from the video clip
    audio = video.audio

    # Write the audio to a WAV file
    audio.write_audiofile(output_file, codec='pcm_s16le')

    print("Conversion completed successfully!")


convert_to_wav(input_file, output_file)
