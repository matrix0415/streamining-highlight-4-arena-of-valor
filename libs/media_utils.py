import os
import  shutil
import requests
from PIL import Image
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def download_file(url, save_path):
    try:
        print("Downloading {} ...".format(save_path.split('/')[-1]))
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            del response
            return [True, save_path]
        else:
            raise Exception("Access Denied, %s" % url)
    except Exception as e:
        print("Error: ", e)
        return [False, e]


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


def video_subclip_ffmpeg(filename, t1, t2, targetname=None, reencode=False):
    """ makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    if not reencode:
        ffmpeg_extract_subclip(filename=filename, t1=t1, t2=t2, targetname=targetname)
    else:
        name, ext = os.path.splitext(filename)
        if not targetname:
            T1, T2 = [int(1000 * t) for t in [t1, t2]]
            targetname = name + "%sSUB%d_%d.%s" % (name, T1, T2, ext)

        cmd = [get_setting("FFMPEG_BINARY"), "-y",
               "-i", filename,
               "-ss", "%0.2f" % t1,
               "-t", "%0.2f" % (t2 - t1),
               "-c:v", "libx264",
               "-c:a", "aac",
               "-b:a", "64k",
               "-strict", "experimental",
               targetname]
        subprocess_call(cmd)


def concatenate_videos_ffmpeg(video_folder, target_file):
    if not os.path.isdir(video_folder):
        assert ValueError, "Video Folder is not a folder. " + video_folder
    video_paths = ["file '%s'" % i for i in sorted(os.listdir(video_folder))]
    with open(os.path.join(video_folder, 'list.txt'), 'w') as f:
        f.write("\n".join(video_paths))
    cmd = [get_setting("FFMPEG_BINARY"), "-y",
           "-f", "concat",
           "-i", os.path.join(video_folder, "list.txt"),
           "-c", "copy", target_file]
    subprocess_call(cmd)
