"""
This is the main file to execute the Whisper Inference and SRT validation process.

All rights reserved.
"""

import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from consolidate.inference import WhisperInference
from consolidate.logger import logger
from consolidate.srt_validation import calculate_error_rate
from consolidate.validator import validate_file

def main():
    """
    Main function to handle command-line inputs for audio and optional SRT file paths.
    """
    parser = argparse.ArgumentParser(description="Process an audio file and an optional subtitle file.")
    
    # Define arguments
    parser.add_argument(
        "--audio_file", 
        required=True, 
        help="Path to the audio file (.mp3, .wav)",
    )
    parser.add_argument(
        "--srt_file", 
        required=False, 
        help="Path to the subtitle file (.srt) [Optional]",
    )

    # Parse the arguments
    args = parser.parse_args()
    audio_file_path = args.audio_file
    srt_file_path = args.srt_file

    try:
        # Validate the audio file
        audio_file = validate_file(audio_file_path, ['.mp3', '.wav'], 'audio file')
        logger.info(f"Audio file validated: {audio_file}")
        print(f"Audio file validated: {audio_file}")

        # Validate the SRT file if provided
        if srt_file_path:

            srt_file = validate_file(srt_file_path, ['.srt'], 'subtitle file')
            logger.info(f"SRT file validated: {srt_file}")
            print(f"SRT file validated: {srt_file}")
        else:
            logger.info("No SRT file provided. Proceeding with audio file only.")
            print("No SRT file provided. Proceeding with audio file only.")

        # Further processing can go here
        logger.info("Files validated successfully. Proceeding with further tasks...")
        print("Files validated successfully. Proceeding with further tasks...")

        # Run inference on audio file
        result = WhisperInference().run_inference(audio_file)
        
        # Save the result to a JSON file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        json_path = f"{base_name}_result.json"

        logger.info(f"Saving the result to JSON file: {json_path}")
        print(f"Saving the result to JSON file: {json_path}")

        # # create a result.json file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        if srt_file_path:
            calculate_error_rate(srt_file=srt_file, json_file=json_path, base_name=base_name)
        logger.info("Inference completed successfully.")

    except ValueError as e:
        logger.error(e)
        print(e)
        exit(1)

if __name__ == "__main__":
    logger.info("Script started")
    try:
        main()
        logger.info("Script completed successfully")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        print(f"An unexpected error occurred: {e}")
