# test videos
cd ~/cv-building-timelapse/data/adjust_translated_rotated/v6_preds_v3/
ffmpeg -framerate 10 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p test_vid_f10_v1.mp4