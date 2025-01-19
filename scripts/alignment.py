"""
This file contains the logic for the alignment and evaluation of a predicted transcript with a ground truth text.

All rights reserved.
"""
import logging
import re
from typing import Dict, List, Tuple

import numpy as np
import pysrt
from Levenshtein import distance as levenshtein_distance
from phonetics import metaphone
from jiwer import remove_punctuation
from transformers import AutoModel, AutoTokenizer

from utils import (AlignmentResult, TimeAlignedSentence, WordTimestamp,
                   fix_json_file, generate_srt_file_output,
                   load_segments_from_json, read_ground_truth_file,
                   read_srt_file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptAligner:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.similarity_threshold = 0.6

    def preprocess_gt_text(self, gt_text: str) -> List[str]:
        """
        Convert text to lowercase and split into sentences.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of preprocessed sentences.
        """
        # use jiwer to remove punctuations, and split into words
        gt_text = remove_punctuation(gt_text)
        return [word.strip().lower() for word in gt_text.strip().split()]

    def get_word_embeddings(self, words: List[str]) -> np.ndarray:
        """
        Convert words to embeddings using a transformer model.

        Args:
            words (List[str]): List of words.

        Returns:
            np.ndarray: Array of word embeddings.
        """
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Use the mean of the last hidden state as the word embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding.
            embedding2 (np.ndarray): Second embedding.

        Returns:
            float: Cosine similarity score.
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity using metaphone encoding.

        Args:
            word1 (str): First word.
            word2 (str): Second word.

        Returns:
            float: Phonetic similarity score.
        """
        return 1.0 if metaphone(word1) == metaphone(word2) else 0.0

    def combined_similarity(
        self, gt_word: str, pred_word: str, gt_embedding: np.ndarray, pred_embedding: np.ndarray
    ) -> float:
        """
        Combine cosine similarity, Levenshtein distance, and phonetic similarity.

        Args:
            gt_word (str): Ground truth word.
            pred_word (str): Predicted word.
            gt_embedding (np.ndarray): Ground truth word embedding.
            pred_embedding (np.ndarray): Predicted word embedding.

        Returns:
            float: Combined similarity score.
        """
        cosine_sim = self.cosine_similarity(gt_embedding, pred_embedding)
        lev_dist = levenshtein_distance(gt_word, pred_word)
        phonetic_sim = self.phonetic_similarity(gt_word, pred_word)

        # Normalize Levenshtein distance to a similarity score
        max_len = max(len(gt_word), len(pred_word))
        lev_sim = 1 - (lev_dist / max_len) if max_len > 0 else 0

        # Weighted combination of similarities
        return 0.5 * phonetic_sim + 0.5 * lev_sim

    def dtw_align_with_embeddings(
        self, ground_truth_text: str, predicted_sentences: List[TimeAlignedSentence]
    ) -> Tuple[List[AlignmentResult], set]:
        """
        Align ground truth text with time-aligned predicted transcript using combined similarity.

        Args:
            ground_truth_text (str): Ground truth text.
            predicted_sentences (List[TimeAlignedSentence]): List of predicted sentences with time alignment.

        Returns:
            Tuple[List[AlignmentResult], set]: List of alignment results and set of matched words.
        """
        # Preprocess ground truth
        gt_words = self.preprocess_gt_text(ground_truth_text)
        pred_sentences = [s.sentence.lower() for s in predicted_sentences]

        # Initialize sliding window parameters
        best_alignment = []
        matched_words = set()
        updated_pred_sentences = []

        # Slide the window over the predicted sentences
        for j, pred_sentence in enumerate(pred_sentences):

            pred_words = pred_sentence.split()
            pred_word_embeddings = self.get_word_embeddings(pred_words)
            max_similarity = -np.inf
            best_segment_alignment = []

            gt_word_embeddings = self.get_word_embeddings(gt_words)
            last_k = 0
            for k in range(len(gt_words) - len(pred_words) + 1):
                # Skip if the current segment is already matched
                if " ".join(gt_words[k : k + len(pred_words)]) in " ".join(matched_words):
                    continue

                # Calculate similarity for the current segment
                # If the segment is in middle, try to find the best match by including the last word of previous segment
                if k > 0 and k < len(gt_words) - len(pred_words):
                    similarity = sum(
                        self.combined_similarity(
                            gt_words[k - 1 + m],
                            pred_words[m],
                            gt_word_embeddings[k - 1 + m],
                            pred_word_embeddings[m],
                        )
                        for m in range(len(pred_words))
                    ) / len(pred_words)

                elif k == 0:
                    similarity = sum(
                        self.combined_similarity(
                            gt_words[k + m], pred_words[m], gt_word_embeddings[k + m], pred_word_embeddings[m]
                        )
                        for m in range(len(pred_words))
                    ) / len(pred_words)
                # If the segment is in the end, try to find the best match by including the last word of previous segment
                else:
                    similarity_lastk_1 = sum(
                        self.combined_similarity(
                            gt_words[k - 1 + m],
                            pred_words[m],
                            gt_word_embeddings[k - 1 + m],
                            pred_word_embeddings[m],
                        )
                        for m in range(len(pred_words))
                    ) / len(pred_words)
                    similarity_lastk = sum(
                        self.combined_similarity(
                            gt_words[k + m], pred_words[m], gt_word_embeddings[k + m], pred_word_embeddings[m]
                        )
                        for m in range(len(pred_words))
                    ) / len(pred_words)
                    similarity = max(similarity_lastk_1, similarity_lastk)
                    if similarity_lastk_1 > similarity_lastk:
                        last_k = k - 1
                    else:
                        last_k = k

                # Find and store segment with maximum similarity
                if similarity > max_similarity:
                    max_similarity = similarity
                    # Same logic as above to find the best match
                    if k > 0 and k < len(gt_words) - len(pred_words):
                        best_segment_alignment.append(
                            (j, gt_words[k - 1 : k + len(pred_words) - 1], max_similarity)
                        )
                    elif k == 0:
                        best_segment_alignment.append((j, gt_words[k : k + len(pred_words)], max_similarity))
                    else:
                        if last_k == k - 1:
                            best_segment_alignment.append(
                                (j, gt_words[k - 1 : k + len(pred_words) - 1], max_similarity)
                            )
                        else:
                            best_segment_alignment.append((j, gt_words[k : k + len(pred_words)], max_similarity))

            # Find the best alignment for the current predicted sentence
            best_sentence_alignment = max(best_segment_alignment, key=lambda x: x[2])
            # Update already matched words
            matched_words.update(best_sentence_alignment[1])

            # If the similarity score is greater than the threshold, add the alignment result
            if best_sentence_alignment and best_sentence_alignment[2] >= self.similarity_threshold:
                best_alignment.append(
                    AlignmentResult(
                        ground_truth_sentence=" ".join(best_sentence_alignment[1]),
                        predicted_sentence=predicted_sentences[j],
                        similarity_score=best_sentence_alignment[2],
                    )
                )
                # Create updated predicted sentence with aligned words
                updated_pred_sentence = TimeAlignedSentence(
                    sentence=predicted_sentences[j].sentence,
                    start_time=predicted_sentences[j].start_time,
                    end_time=predicted_sentences[j].end_time,
                    words=[
                        WordTimestamp(
                            word=gt_word,
                            start=predicted_sentences[j].words[m].start,
                            end=predicted_sentences[j].words[m].end,
                        )
                        for m, gt_word in enumerate(best_sentence_alignment[1])
                    ],
                )
                updated_pred_sentences.append(updated_pred_sentence)

        return best_alignment, matched_words, updated_pred_sentences


    def post_process_results(
        self, results: List[AlignmentResult], ground_truth_text: str, matched_words: set
    ) -> List[AlignmentResult]:
        """
        Post-process the alignment results to handle unaligned and overlapping words.

        Args:
            results (List[AlignmentResult]): List of alignment results.
            ground_truth_text (str): Ground truth text.
            matched_words (set): Set of matched words.

        Returns:
            List[AlignmentResult]: List of post-processed alignment results.
        """
        gt_words_unaligned = set(ground_truth_text.split()) - matched_words

        for i in range(1, len(results)):
            prev_result = results[i - 1]
            curr_result = results[i]

            if prev_result.predicted_sentence and curr_result.predicted_sentence:
                prev_end_time = prev_result.predicted_sentence.end_time
                curr_start_time = curr_result.predicted_sentence.start_time

                if prev_end_time == curr_start_time:
                    prev_words = set(prev_result.ground_truth_sentence)
                    curr_words = set(curr_result.ground_truth_sentence)

                    #########################

                    #########################
                    # Form a string which is supposed to match with ground truth
                    match_string = " ".join(prev_words) + " " + " ".join(curr_words)
                    if match_string in ground_truth_text:
                        continue
                    else:
                        # Identify unaligned words in the between the segments
                        for unaligned_word in gt_words_unaligned:
                            match_string = " ".join(prev_words) + " " + unaligned_word + " " + " ".join(curr_words)
                            if match_string in ground_truth_text:
                                # Add them to the previous segment
                                prev_result.ground_truth_sentence.append(unaligned_word)
                                gt_words_unaligned.remove(unaligned_word)
                                break

                        # Identify overlapping words
                        overlapping_words = prev_words & curr_words
                        # Remove overlapping words from the previous segment
                        if overlapping_words:
                            prev_result.ground_truth_sentence = [
                                word for word in prev_result.ground_truth_sentence if word not in overlapping_words
                            ]

                    # Replace predicted words with matched ground truth words
                    prev_result.predicted_sentence.words = [
                        word
                        if word.word not in prev_words
                        else WordTimestamp(
                            word=prev_result.ground_truth_sentence.split()[list(prev_words).index(word.word)],
                            start=word.start,
                            end=word.end,
                        )
                        for word in prev_result.predicted_sentence.words
                    ]
                    curr_result.predicted_sentence.words = [
                        word
                        if word.word not in curr_words
                        else WordTimestamp(
                            word=curr_result.ground_truth_sentence.split()[list(curr_words).index(word.word)],
                            start=word.start,
                            end=word.end,
                        )
                        for word in curr_result.predicted_sentence.words
                    ]

        return results



    def calculate_subtitle_error_rate(self, word_srt_path, sentence_srt_path):
        """
        Aligns word-level timestamps from one SRT file with sentences in another,
        identifying potential timestamp errors.

        Args:
            word_srt_path (str): Path to the SRT file with word-level timestamps.
            sentence_srt_path (str): Path to the SRT file with sentences.
        """
        try:
            word_subs = pysrt.open(word_srt_path)
            sentence_subs = pysrt.open(sentence_srt_path)
        except Exception as e:
            print(f"Error opening SRT files: {e}")
            return

        aligned_results = []
        error = []
        min_ts = float("inf")
        max_ts = float("-inf")

        matched_word_indices = set()
        L = 0

        deviation = 0
        for sentence_sub in sentence_subs:
            # Regex to match any character that is not a word character (alphanumeric + underscore) or whitespace
            # pattern = r'[^\w\s]'

            # Substitute matched characters with empty string
            # sentence_text = re.sub(pattern, '', sentence_sub.text)

            sentence_text = " ".join([sub.strip() for sub in sentence_sub.text.split()])

            # sentence_text = sentence_sub.text
            L = len(sentence_text.split())
            print(f"Sentence Text: {sentence_text}")
            print(f"Sentence Length : {L}")

            # Find corresponding word-level timestamps by joining adjacent words
            start_time = float("inf")
            end_time = float("-inf")

            new_matched_word_indices = set()
            # Join adjacent word sub texts and find matches
            # joined_text = ''
            joined_sub = []
            for idx, word_sub in enumerate(word_subs):
                if idx in matched_word_indices:
                    continue  # Skip already matched word_subs

                word = word_sub.text
                joined_sub.append(word_sub)
                new_matched_word_indices.add(idx)

                # Check if the joined text matches the sentence text
                # Remove punctuations from joined text
                # cleaned_joined_text = re.sub(pattern, '', joined_text)

                if len(joined_sub) <= L + 1:
                    joined_text = " ".join([sub.text for sub in joined_sub])
                    print(f"Joined text: {joined_text}")
                    # print(f"Joined Length : {len(joined_text.split())}")
                    if sentence_text == joined_text:
                        print(f"Matched: {joined_text}")
                        start_time = min(start_time, joined_sub[0].start.ordinal / 1000)
                        end_time = max(end_time, joined_sub[-1].end.ordinal / 1000)
                        matched_word_indices.update(new_matched_word_indices)

                        new_matched_word_indices = set()
                        # joined_text = ''
                        joined_sub = []
                        break
                else:
                    # Removing the first word in joined_text, as the length has exceeded
                    joined_sub = joined_sub[1:]
                    new_matched_word_indices.remove(idx - L)

            min_ts = min(sentence_sub.start.ordinal, min_ts)
            max_ts = max(sentence_sub.end.ordinal, max_ts)

            # Calculate deviation
            if start_time != float("inf"):
                deviation = abs(start_time - sentence_sub.start.ordinal / 1000)
                deviation = max(0, deviation - 0.5)
                error.append(deviation)
                print(deviation)

        total_length = (max_ts - min_ts) / 1000 if max_ts > min_ts else 1  # Avoid division by zero

        # Calculate Subtitle Error Rate (SER)
        ser = (sum(error) / total_length) * 100 if total_length > 0 else 0
        print(f"Subtitle Error Rate (SER): {ser:.3f}%")


if __name__ == "__main__":
    print("Loading input predicted srt file ...\n")
    # Load Input srt file
    # pred_srt_sentences = read_srt_file('/home/arjun/naren/subtitle_alignment/data/test/Abdul_Kalam,_Designing_a_Fighter_Jet_Punjabi.srt')
    input_json_path = "/home/arjun/naren/alignment/scripts/segments.json"
    output_json_path = "/home/arjun/naren/alignment/scripts/segments_fixed.json"
    fix_json_file(input_json_path, output_json_path)
    print(f"Fixed JSON written to {output_json_path}")
    pred_srt_sentences = load_segments_from_json("/home/arjun/naren/alignment/scripts/segments_fixed.json")
    print("Loading ground truth text file ...\n")
    # Load Ground Truth text file
    ground_truth_text = read_ground_truth_file(
        "/home/arjun/naren/alignment/data/Punjabi/Text/AniBook Text/What_Should_Soma_Grow_Punjabi.txt"
    )

    print("Starting alignment process ...\n")
    # Align the predicted sentences with the ground truth text
    aligner = TranscriptAligner()
    results, matched_words, updated_pred_sentences = aligner.dtw_align_with_embeddings(
        ground_truth_text, pred_srt_sentences
    )
    print("Obtained all the results\n\n")

    print("Post processing the results ...\n")
    # Post processing
    results = post_process_results(results, ground_truth_text, matched_words)

    print("Alignment Results:")
    print("-" * 60)
    print("   Ground Truth  |  Predicted  |  Time  | Similarity   ")
    print("-" * 60)

    for result, updated_pred in zip(results, updated_pred_sentences):
        if result.predicted_sentence:
            print(
                f"{' '.join(result.ground_truth_sentence):<50} | "
                f"{result.predicted_sentence.sentence:<50} | "
                f"{result.predicted_sentence.start_time:.2f}-{result.predicted_sentence.end_time:.2f} | "
                f"{result.similarity_score:.3f}"
            )
            print("Word Level Timestamps for this predicted segment:")
            for word in updated_pred.words:
                print(f"   {word.word:<15} | Start: {word.start:.2f} | End: {word.end:.2f}")
            print("-" * 60)
        else:
            print(f"{result.ground_truth_sentence:<50} | {'':50} | {'':8} | {result.similarity_score:.3f}")

    generate_srt_file_output(
        results,
        updated_pred_sentences,
        "/home/arjun/naren/alignment/data/Punjabi/test/Aligned_transcript_word_level.srt",
        "/home/arjun/naren/alignment/data/Punjabi/test/Aligned_transcript_sentence_level.srt",
    )
    print("Aligned GT transcript saved as Aligned_GT_transcript.srt")

    # Replace placeholders with actual file paths
    calculate_subtitle_error_rate(
        "/home/arjun/naren/alignment/data/test/Aligned_transcript_word_level.srt",
        "/home/arjun/naren/alignment/data/Punjabi/Ground Truth SRT/AniBook SRT/What_did_you_see_Punjabi.srt",
    )
