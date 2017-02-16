from moviepy.editor import VideoFileClip
from image_processing import process_image


def process_video(video_path, output_path='project_video_out.mp4'):
    """
    Takes an input video of lane lines and saves a new video with the lane lanes, curvature radius and camera offset from
    center annotated.
    :param video_path: Input video location.
    :param output_path: Output video location.
    """
    # Load the original video
    input_video = VideoFileClip(video_path)

    # Process and save
    processed_video = input_video.fl_image(process_image)
    processed_video.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    input_video_path = 'project_video.mp4'
    process_video(input_video_path)
