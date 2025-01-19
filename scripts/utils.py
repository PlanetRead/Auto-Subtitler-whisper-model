"""
This file contains the utility functions for subtitle alignment.

All rights reserved.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import List, Tuple, Dict


import pysrt
from pydub import AudioSegment


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TimeAlignedSentence:
    sentence: str
    start_time: float
    end_time: float
    words: List[WordTimestamp]


@dataclass
class AlignmentResult:
    ground_truth_sentence: list[str]
    predicted_sentence: TimeAlignedSentence | None
    similarity_score: float


def read_srt_file(srt_file_path: str) -> List[TimeAlignedSentence]:
    """
    Reads an SRT file and converts each subtitle to a TimeAlignedSentence object.

    Args:
        srt_file_path (str): Path to the SRT file.

    Returns:
        List[TimeAlignedSentence]: List of time-aligned sentences.
    """
    # Read the SRT file
    subs = pysrt.open(srt_file_path)

    # Convert each subtitle to TimeAlignedSentence
    sentences = []
    for sub in subs:
        sentence = TimeAlignedSentence(
            sentence=sub.text,
            start_time=sub.start.seconds + sub.start.minutes * 60 + sub.start.hours * 3600,
            end_time=sub.end.seconds + sub.end.minutes * 60 + sub.end.hours * 3600
        )
        sentences.append(sentence)

    return sentences



def srt_to_text(srt_file_path, output_text_path=None):
    """
    Convert an SRT subtitle file to plain text using pysrt library.
    Handles Punjabi text with multiple spaces between words and preserves line breaks.
    
    Args:
        srt_file_path (str): Path to the input SRT file
        output_text_path (str, optional): Path for the output text file. 
            If None, will use the same name as input with .txt extension
    
    Returns:
        str: Path to the created text file
    """
    if output_text_path is None:
        output_text_path = srt_file_path.rsplit(".", 1)[0] + ".txt"

    # Load the SRT file
    subs = pysrt.open(srt_file_path, encoding="utf-8")

    # Extract text from each subtitle
    text_lines = [" ".join(sub.text.split()) for sub in subs]

    # Write the extracted text to the output file
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(text_lines))

    return output_text_path


def convert_srt_directory(input_dir: str, output_dir: str):
    """
    Convert all SRT files in input_dir (including subdirectories) to text files,
    preserving the directory structure in output_dir.
    
    Args:
        input_dir (str): Path to input directory containing SRT files
        output_dir (str): Path where text files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .srt files in input directory and subdirectories
    for srt_file in input_path.rglob("*.srt"):
        # Get the relative path from input directory
        rel_path = srt_file.relative_to(input_path)

        # Construct output path with same directory structure
        output_file = output_path / rel_path.with_suffix(".txt")

        # Create necessary subdirectories
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert the file
        srt_to_text(str(srt_file), str(output_file))
        print(f"Converted: {srt_file.name}")


def split_audio_and_srt_for_finetuning(audio_dir: str, srt_dir: str, output_dir: str):
    """
    Split audio files based on SRT timestamps into segments of 30 seconds or less.
    Combines consecutive subtitles if their total duration is under 30 seconds.
    
    Args:
        audio_dir (str): Directory containing audio files
        srt_dir (str): Directory containing corresponding SRT files
        output_dir (str): Directory to save split audio and transcript files
    """

    audio_path = Path(audio_dir)
    srt_path = Path(srt_dir)
    output_path = Path(output_dir)

    # Create output directories
    audio_output = output_path / "audio"
    text_output = output_path / "text"
    audio_output.mkdir(parents=True, exist_ok=True)
    text_output.mkdir(parents=True, exist_ok=True)

    # Process each audio file
    for audio_file in audio_path.rglob("*.wav"):  # Adjust extension if needed
        # Find corresponding SRT file
        srt_file = srt_path / f"{audio_file.stem}.srt"
        if not srt_file.exists():
            print(f"No SRT file found for {audio_file.name}")
            continue

        # Load audio and subtitles
        audio = AudioSegment.from_file(str(audio_file))
        subs = pysrt.open(str(srt_file))

        # Group subtitles into 30-second chunks
        current_chunk = []
        current_duration = 0
        chunk_index = 0

        for sub in subs:
            start_ms = (
                sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds
            ) * 1000 + sub.start.milliseconds
            end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
            duration = end_ms - start_ms

            if current_duration + duration <= 30000:  # 30 seconds in milliseconds
                current_chunk.append(sub)
                current_duration += duration
            else:
                # Save current chunk
                if current_chunk:
                    save_chunk(audio, current_chunk, audio_file.stem, chunk_index, audio_output, text_output)
                    chunk_index += 1
                # Start new chunk
                current_chunk = [sub]
                current_duration = duration

        # Save final chunk if any
        if current_chunk:
            save_chunk(audio, current_chunk, audio_file.stem, chunk_index, audio_output, text_output)

        print(f"Processed: {audio_file.name}")


def save_chunk(
    audio: "AudioSegment", subs: list, base_name: str, chunk_index: int, audio_output: Path, text_output: Path
):
    """Helper function to save audio chunk and corresponding text"""
    # Get time bounds
    start_ms = (subs[0].start.hours * 3600 + subs[0].start.minutes * 60 + subs[0].start.seconds) * 1000 + subs[
        0
    ].start.milliseconds
    end_ms = (subs[-1].end.hours * 3600 + subs[-1].end.minutes * 60 + subs[-1].end.seconds) * 1000 + subs[
        -1
    ].end.milliseconds

    # Extract and save audio chunk
    audio_chunk = audio[start_ms:end_ms]
    audio_filename = f"{base_name}_{chunk_index:04d}.wav"
    audio_chunk.export(str(audio_output / audio_filename), format="wav")

    # Save transcript
    text = " ".join(sub.text.replace("\n", " ") for sub in subs)
    text_filename = f"{base_name}_{chunk_index:04d}.txt"
    with open(text_output / text_filename, "w", encoding="utf-8") as f:
        f.write(text)



def load_segments_from_json(json_path: str) -> List[TimeAlignedSentence]:
    with open(json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    time_aligned_sentences = []
    for segment in result["segments"]:
        # Split text on punctuation marks
        text = segment["text"]
        words_data = segment["words"]
        
        # Track current sentence being built
        current_sentence_words = []
        current_words_data = []
        
        for i, word_data in enumerate(words_data):
            current_sentence_words.append(word_data["word"])
            current_words_data.append(word_data)
            
            # Check conditions for splitting:
            # 1. Punctuation marks
            # 2. Length > 5 words
            # 3. Last word in segment
            word = word_data["word"]
            should_split = (
                any(punct in word for punct in ['.', '।', '?']) or
                len(current_sentence_words) > 5 or
                i == len(words_data) - 1
            )
            
            if should_split and current_words_data:
                # Create WordTimestamp objects for current sentence
                word_timestamps = [
                    WordTimestamp(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"]
                    ) for w in current_words_data
                ]
                
                # Create TimeAlignedSentence with accurate timestamps
                time_aligned_sentence = TimeAlignedSentence(
                    sentence=" ".join(current_sentence_words),
                    start_time=current_words_data[0]["start"],
                    end_time=current_words_data[-1]["end"],
                    words=word_timestamps
                )
                
                time_aligned_sentences.append(time_aligned_sentence)
                
                # Reset for next sentence
                current_sentence_words = []
                current_words_data = []

    return time_aligned_sentences


def read_ground_truth_file(file_path: str) -> str:
    """
    Reads the ground truth text file.

    Args:
        file_path (str): Path to the ground truth text file.

    Returns:
        str: Ground truth text.
    """
    with open(file_path, "r") as file:
        ground_truth = file.read()
    return ground_truth


def srt_to_text(srt_file_path: str, output_file_path: str) -> None:
    """
    Converts an SRT file to a text file.

    Args:
        srt_file_path (str): Path to the input SRT file.
        output_file_path (str): Path to the output text file.
    """
    subs = pysrt.open(srt_file_path)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for sub in subs:
            output_file.write(sub.text + "\n")


def generate_srt_file_output(
    alignment_results: List[AlignmentResult],
    updated_pred_sentences: List[TimeAlignedSentence],
    output_file_word_level: str,
    output_file_sentence_level: str,
):
    """
    Generates an SRT file from the alignment results.

    Args:
        alignment_results (List[AlignmentResult]): List of alignment results.
        updated_pred_sentences (List[TimeAlignedSentence]): List of updated predicted sentences with word-level timestamps.
        output_file_word_level (str): Path to the output word-level SRT file.
        output_file_sentence_level (str): Path to the output sentence-level SRT file.
    """
    subs = []
    for i, updated_pred in enumerate(updated_pred_sentences):
        for word in updated_pred.words:
            start_time = word.start
            end_time = word.end
            text = word.word

            sub = pysrt.SubRipItem(
                index=len(subs) + 1, start=seconds_to_srt_time(start_time), end=seconds_to_srt_time(end_time), text=text
            )
            subs.append(sub)

    # Generate sentence level aligned transcript
    sentence_subs = []
    for i, result in enumerate(alignment_results):
        if result.predicted_sentence:
            start_time = result.predicted_sentence.start_time
            end_time = result.predicted_sentence.end_time
            text = " ".join(result.ground_truth_sentence)

            sub = pysrt.SubRipItem(
                index=len(sentence_subs) + 1,
                start=seconds_to_srt_time(start_time),
                end=seconds_to_srt_time(end_time),
                text=text,
            )
            sentence_subs.append(sub)

    # Save the sentence level subtitles to an SRT file
    sentence_subtitle_file = pysrt.SubRipFile(sentence_subs)
    sentence_subtitle_file.save(output_file_sentence_level)

    # Save the subtitles to an SRT file
    subtitle_file = pysrt.SubRipFile(subs)
    subtitle_file.save(output_file_word_level)


def seconds_to_srt_time(seconds: float) -> pysrt.SubRipTime:
    """
    Converts seconds to pysrt.SubRipTime object.

    Args:
        seconds (float): Time in seconds.

    Returns:
        pysrt.SubRipTime: SubRipTime object representing the time.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    second = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return pysrt.SubRipTime(hours, minutes, second, milliseconds)



def fix_json_file(input_path, output_path):
    """
    Reads a JSON file, fixes unquoted keys, and saves it to a new file.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace single quotes with double quotes
    content = content.replace("'", '"')
    # Ensure keys are quoted if they are not already
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON, please inspect {input_path} manually. Error message: {e}")
        return

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)




def combine_repeated_words(srt_file_path, output_file_path):
    # Open the SRT file
    subs = pysrt.open(srt_file_path)
    combined_subs = pysrt.SubRipFile()

    previous_word = None
    start_time = None
    end_time = None

    for sub in subs:
        # Clean the text to avoid whitespace issues
        current_word = sub.text.strip()

        if previous_word is None:
            # Initialize the first word block
            previous_word = current_word
            start_time = sub.start
            end_time = sub.end
        elif current_word == previous_word:
            # If the word is repeated, update the end time
            end_time = sub.end
        else:
            # Add the combined segment to the output
            combined_subs.append(pysrt.SubRipItem(index=len(combined_subs) + 1, start=start_time, end=end_time, text=previous_word))
            # Start a new block for the new word
            previous_word = current_word
            start_time = sub.start
            end_time = sub.end

    # Add the last remaining block
    combined_subs.append(pysrt.SubRipItem(index=len(combined_subs) + 1, start=start_time, end=end_time, text=previous_word))

    combined_subs.save(output_file_path)
    print(f"Combined SRT saved at {output_file_path}")

def split_seconds(seconds: float) -> tuple[int, int]:
    """
    Splits a float value representing seconds into integer seconds and integer milliseconds.

    Args:
        seconds (float): Time in seconds, with up to 3 decimal places.

    Returns:
        tuple[int, int]: A tuple containing (integer seconds, integer milliseconds).
    """
    seconds_int = int(seconds)
    milliseconds = int(float(seconds % 1) * 1000)
    return seconds_int, milliseconds
