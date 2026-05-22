from moviepy import VideoFileClip
import argparse


def to_seconds(timestamp):
    try:
        minutes, seconds = map(int, timestamp.split(':'))
        return minutes * 60 + seconds
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {timestamp}. Use MM:SS")


def cut_video(start_time, end_time, input_path, output_path):
    start_sec = to_seconds(start_time)
    end_sec = to_seconds(end_time)

    if start_sec >= end_sec:
        raise ValueError("Start time must be less than end time.")

    clip = VideoFileClip(input_path).subclipped(start_sec, end_sec)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"Saved cut video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cut a segment from a video.")

    # positional arguments
    parser.add_argument("-start", default="01:48", help="Start timestamp (MM:SS)")
    parser.add_argument("-end", default="03:51", help="End timestamp (MM:SS)")

    # optional arguments
    parser.add_argument("-i", "--input", required=False, default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/GROUND_LEVEL.MP4",
                        help="Path to input video file")
    parser.add_argument("-o", "--output", default="output.mp4",
                        help="Path to output video file (default: output.mp4)")

    args = parser.parse_args()

    cut_video(args.start, args.end, args.input, args.output)


if __name__ == "__main__":
    main()