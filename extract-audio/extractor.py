# extract-audio/extractor.py

import os
from moviepy.editor import VideoFileClip

def extract_audio(video_filename: str, output_format: str = "mp3") -> dict:
    """
    Extracts the audio from a video file and saves it as an audio file.

    Args:
        video_filename (str): Path to the video file (relative or absolute).
        output_format (str): Desired audio format (e.g. 'mp3', 'wav').

    Returns:
        dict: {
            "success": bool,
            "audio_path": str or None,
            "error": str or None
        }
    """

    try:
        # Validate file existence
        if not os.path.exists(video_filename):
            return {"success": False, "audio_path": None, "error": f"File not found: {video_filename}"}

        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(video_filename))[0]

        # Define output directory (same as input file)
        output_dir = os.path.dirname(video_filename)
        audio_output_path = os.path.join(output_dir, f"{base_name}.{output_format}")

        # Load the video
        with VideoFileClip(video_filename) as video:
            if not video.audio:
                return {"success": False, "audio_path": None, "error": "No audio track found in video."}
            
            # Write the audio file
            video.audio.write_audiofile(audio_output_path, logger=None)

        return {"success": True, "audio_path": audio_output_path, "error": None}

    except Exception as e:
        return {"success": False, "audio_path": None, "error": str(e)}


"""
SAMPLE CALL FOR INTEGRATION



# app.py

from flask import Flask, jsonify, request
from extract-audio.extractor import extract_audio
import os

app = Flask(__name__)

@app.route("/extract", methods=["POST"])
def extract_audio_route():
    data = request.json
    video_name = data.get("video_name")

    if not video_name:
        return jsonify({"error": "Missing 'video_name' parameter"}), 400

    video_path = os.path.join("videos", video_name)
    result = extract_audio(video_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)





"""