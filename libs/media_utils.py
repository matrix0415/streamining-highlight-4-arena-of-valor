import os
from PIL import Image
from moviepy.config import get_setting
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.tools import subprocess_call


def video_to_img(video_file, target_path, fps=1):
    if not os.path.isfile(video_file):
        assert ValueError, "Video File doesn't exist. " + video_file
    basename = os.path.basename(video_file)[:-4]
    clip = VideoFileClip(video_file)
    clip.write_images_sequence(nameformat="{}/{}.frame.%06d.jpg".format(target_path, basename), fps=fps)


def video_to_img_ffmpeg(video_file, target_path, fps=1):
    if not os.path.isfile(video_file):
        assert ValueError, "Video File doesn't exist. " + video_file
    basename = os.path.basename(video_file)[:-4]
    t_path = "{}/{}.frame.%06d.jpg".format(target_path, basename)
    cmd = [get_setting("FFMPEG_BINARY"), "-y",
           "-i", video_file,
           "-vf", "fps=%f" % fps,
           t_path]
    subprocess_call(cmd)


def img_to_video(img_folder, target_file, fps=10):
    clip = ImageSequenceClip(img_folder, fps=fps)
    clip.write_videofile(target_file, codec='libx264', threads=4)
    [os.remove(os.path.join(img_folder, i)) for i in os.listdir(img_folder) if i[-4:] != '.mp4']


def resize_img(target_path, target_size=None):
    img_list = []
    if os.path.isfile(target_path):
        img_list = [target_path]
    if os.path.isdir(target_path):
        img_list = [os.path.join(target_path, i) for i in os.listdir(target_path)]
    # for smaller file size.
    for p in img_list:
        if '.jpg' in p:
            img = Image.open(p)
            img = img.resize(target_size or (int(img.size[0] * 0.7), int(img.size[1] * 0.7)), Image.ANTIALIAS)
            img.save(p)


