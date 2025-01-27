import json
from typing import List
import pysrt

from consolidate.logger import logger
from consolidate.schemas import AlignmentResult, TimeAlignedSentence, WordTimestamp

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
    logger.info(f"Saving sentence level aligned transcript to {output_file_sentence_level}")
    sentence_subtitle_file.save(output_file_sentence_level)

    # Save the subtitles to an SRT file
    subtitle_file = pysrt.SubRipFile(subs)
    logger.info(f"Saving word level aligned transcript to {output_file_word_level}")
    subtitle_file.save(output_file_word_level)

def calculate_subtitle_error_rate(word_srt_path, sentence_srt_path, tolerance_threshold=0.1, max_deviation=5):
    """
    Calculates Subtitle Error Rate (SER) by comparing word-level and sentence-level SRT files.
    
    Args:
        word_srt_path (str|Path): Path to the SRT file with word-level timestamps
        sentence_srt_path (str|Path): Path to the SRT file with sentence-level timestamps
        tolerance_threshold (float): Acceptable timestamp deviation threshold in seconds (default: 0.1)
        max_deviation (float): Maximum allowed deviation before excluding from calculation (default: 5)
    
    Returns:
        dict: Dictionary containing:
            - ser: Subtitle Error Rate as percentage
            - deviations: List of timestamp deviations
            - matches: Number of successful matches
            - total_sentences: Total number of sentences processed
    """
    try:
        word_subs = pysrt.open(str(word_srt_path))
        sentence_subs = pysrt.open(str(sentence_srt_path))
    except Exception as e:
        logger.error(f"Error opening SRT files: {e}")
        raise RuntimeError(f"Error opening SRT files: {e}")

    # Initialize tracking variables
    deviations = []
    matches = 0
    total_sentences = len(sentence_subs)
    matched_word_indices = set()
    
    # Track overall timeline
    timeline = {
        'start': min(sub.start.ordinal for sub in sentence_subs),
        'end': max(sub.end.ordinal for sub in sentence_subs)
    }

    def clean_text(text):
        """Normalize text for comparison"""
        return " ".join(word.strip() for word in text.split())

    def find_matching_words(sentence_text, word_subs, matched_indices, max_attempts=5):
        """Find matching words for a sentence with sliding window"""
        sentence_length = len(sentence_text.split())
        window = []
        new_matches = set()

        for idx, word_sub in enumerate(word_subs):
            if idx in matched_indices:
                continue

            window.append(word_sub)
            new_matches.add(idx)

            # Remove words from start if window too large
            while len(window) > sentence_length:
                window.pop(0)
                new_matches.remove(min(new_matches))

            if len(window) == sentence_length:
                window_text = clean_text(" ".join(sub.text for sub in window))
                if window_text == sentence_text:
                    return {
                        'start': window[0].start.ordinal / 1000,
                        'end': window[-1].end.ordinal / 1000,
                        'matches': new_matches
                    }

        return None

    # Process each sentence
    for sentence_sub in sentence_subs:
        sentence_text = clean_text(sentence_sub.text)
        sentence_start = sentence_sub.start.ordinal / 1000

        # Find matching words
        match_result = find_matching_words(sentence_text, word_subs, matched_word_indices)
        
        if match_result:
            matches += 1
            matched_word_indices.update(match_result['matches'])
            
            # Calculate deviation
            deviation = abs(match_result['start'] - sentence_start)
            if deviation > tolerance_threshold and deviation <= max_deviation:
                deviations.append(deviation)

    # Calculate final metrics
    total_duration = (timeline['end'] - timeline['start']) / 1000
    ser = (sum(deviations) / total_duration * 100) if total_duration > 0 else 0

    results = {
        'ser': round(ser, 3),
        'deviations': deviations,
        'matches': matches,
        'total_sentences': total_sentences,
        'match_rate': round(matches / total_sentences * 100, 2) if total_sentences > 0 else 0
    }

    # Print summary
    logger.info(f"Subtitle Error Rate (SER): {results['ser']}%")
    print(f"Subtitle Error Rate (SER): {results['ser']}%")
    logger.info(f"Match Rate: {results['match_rate']}% ({matches}/{total_sentences} sentences)")
    print(f"Match Rate: {results['match_rate']}% ({matches}/{total_sentences} sentences)\n")
    logger.info(f"Average Deviation: {round(sum(deviations) / len(deviations), 3) if deviations else 0} seconds")
    print(f"Average Deviation: {round(sum(deviations) / len(deviations), 3) if deviations else 0} seconds \n")

    return results


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
    with open(file_path, "r", encoding="utf-8") as file:
        ground_truth = file.read()
    return ground_truth