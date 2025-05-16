import os

import imageio
from moviepy.editor import VideoFileClip


def export_gif(config_name, frames, gif_save_path, config_data, timestamp_str, reward):
    gif_folder = os.path.join(gif_save_path, f"<{reward}>{timestamp_str}_{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    gif_path = os.path.join(
        gif_folder,
        f"multiwalker_{config_name}.gif",
    )
    imageio.mimwrite(
        gif_path,
        frames,
        duration=10,
    )

    clip = VideoFileClip(gif_path)
    clip.write_videofile(
        os.path.join(
            gif_folder,
            f"<{reward}>multiwalker_{config_name}.mp4",
        ),
        codec="libx264",
        logger=None,
    )
