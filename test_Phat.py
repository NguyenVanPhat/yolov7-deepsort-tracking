fps_video_src = 30

origin_fps = 20
origin_frame_throughout = 3
stride_fps = 5
stride_frame_throughout = 2
if int(fps_video_src) < 20:
    number_frame_throughout = 2
elif int(fps_video_src) == origin_fps:
    number_frame_throughout = int(origin_frame_throughout)
else:
    number_frame_throughout = int(origin_frame_throughout + (stride_frame_throughout*((int(fps_video_src)-origin_fps)/stride_fps)))
print(number_frame_throughout)