import os
import shutil

from pathlib import Path

def copy_videos(source_dir ,dest_dir, format = "mp4"):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    for path in Path(source_dir).rglob(f'*.{format}'):
        shutil.copy(path , dest_dir)


if __name__ == "__main__":
    src = "F:/Data/drone_images/images/images"
    dest_dir = "F:/Data/drone_images/src_videos"
    copy_videos(src , dest_dir)
