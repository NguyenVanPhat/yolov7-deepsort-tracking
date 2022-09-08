origin_fps = 20
stride_fps = 5
fps_video_src = 30.0

origin_frame_throughout = 2
stride_frame_throughout = 3

print(int(origin_frame_throughout + (stride_frame_throughout*((int(fps_video_src)-origin_fps)/stride_fps))))