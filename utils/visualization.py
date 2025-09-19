import cv2

def frames2mp4(frame_paths, output_path, fps):
    # prepare video writer
    frame = cv2.imread(frame_paths[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # write
    for p in frame_paths:
        frame = cv2.imread(p)
        video.write(frame)
    video.release()