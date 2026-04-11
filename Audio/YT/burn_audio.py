from moviepy.editor import *

output_path = "/home/kyue/Videos/pri/miaomiaozi_urine.mp4"
video_path = "/home/kyue/Downloads/sw/out.mp4"
final_output_path = "/home/kyue/Downloads/sw/pangding.mp4"

video = VideoFileClip(output_path)
audio = VideoFileClip(video_path)

video_with_audio = video.set_audio(audio.audio)
video_with_audio.write_videofile(final_output_path)