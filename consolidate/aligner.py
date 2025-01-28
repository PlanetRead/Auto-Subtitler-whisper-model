"""
This file contains the logic for the alignment and evaluation of a predicted transcript with a ground truth text.

All rights reserved.
"""


import numpy as np
from typing import List, Tuple
from Levenshtein import distance as levenshtein_distance
from phonetics import metaphone
from consolidate.schemas import WordTimestamp, TimeAlignedSentence, AlignmentResult

class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        return " ".join([word.strip() for word in text.strip().split()])

class SimilarityCalculator:
    def __init__(self, similarity_threshold=0.75):
        self.similarity_threshold = similarity_threshold

    def phonetic_similarity(self, word1: str, word2: str) -> float:
        return 1.0 if metaphone(word1) == metaphone(word2) else 0.0

    def levenshtein_similarity(self, word1: str, word2: str) -> float:
        lev_dist = levenshtein_distance(word1, word2)
        max_len = max(len(word1), len(word2))
        return 1 - (lev_dist / max_len) if max_len > 0 else 0

    def combined_similarity(self, word1: str, word2: str) -> float:
        phonetic_sim = self.phonetic_similarity(word1, word2)
        lev_sim = self.levenshtein_similarity(word1, word2)
        return 0.5 * phonetic_sim + 0.5 * lev_sim

class EnhancedTranscriptAligner:
    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold
        self.similarity_calculator = SimilarityCalculator(similarity_threshold)

    def align_predicted_sentences(
        self,
        ground_truth: str,
        predicted_sentences: List[TimeAlignedSentence]
    ) -> Tuple[List[AlignmentResult], set, List[TimeAlignedSentence]]:
        processed_gt_text = TextPreprocessor.preprocess_text(ground_truth)
        gt_words = processed_gt_text.split()

        best_alignment = []
        matched_word_indices = set()
        updated_pred_sentences = []
        start_search_position = 0

        dp_matrix = {}

        for p, pred_sentence in enumerate(predicted_sentences):
            pred_words = pred_sentence.sentence.lower().split()
            if not pred_words:
                continue

            window_size = len(pred_words)
            best_score = -np.inf
            best_position = None

            search_positions = []
            start_search_position += len(pred_words)
            search_positions.extend(range(start_search_position, min(start_search_position + 5, len(gt_words) - window_size + 1)))
            search_positions.extend(range(max(0, start_search_position - 5), start_search_position))

            for k in search_positions:
                if any(idx in matched_word_indices for idx in range(k, k + window_size)):
                    continue

                if k + window_size > len(gt_words):
                    continue

                score = 0
                for m in range(window_size):
                    pair_key = (gt_words[k + m], pred_words[m])
                    if pair_key not in dp_matrix:
                        dp_matrix[pair_key] = self.similarity_calculator.combined_similarity(*pair_key)
                    score += dp_matrix[pair_key]
                score /= window_size

                if score > best_score:
                    best_score = score
                    best_position = k

            if best_position is not None and best_score >= self.similarity_calculator.similarity_threshold:
                print(f"Best alignment found at position {best_position} with score {best_score}")
                aligned_indices = range(best_position, best_position + window_size)
                matched_word_indices.update(aligned_indices)

                aligned_words = [gt_words[idx] for idx in aligned_indices]
                gt_sentence = ' '.join(aligned_words)
                print(f"Aligned GT sentence: {gt_sentence}\n")

                context_bonus = 0
                if best_alignment:
                    last_end = max(matched_word_indices - set(aligned_indices))
                    if best_position == last_end + 1:
                        context_bonus = 0.1
                    elif abs(best_position - last_end) <= 3:
                        context_bonus = 0.05

                final_score = min(1.0, best_score + context_bonus)

                try:
                    result = AlignmentResult(
                        ground_truth_sentence=gt_sentence,
                        predicted_sentence=pred_sentence,
                        similarity_score=final_score
                    )
                    best_alignment.append(result)

                    updated_pred = TimeAlignedSentence(
                        sentence=gt_sentence,
                        start_time=pred_sentence.start_time,
                        end_time=pred_sentence.end_time,
                        words=[
                            WordTimestamp(
                                word=gt_word,
                                start=pred_sentence.words[m].start,
                                end=pred_sentence.words[m].end
                            )
                            for m, gt_word in enumerate(aligned_words)
                        ]
                    )
                    updated_pred_sentences.append(updated_pred)
                except Exception as e:
                    print(f"Error creating updated predicted sentence: {e}")
                    continue

        print(f"Aligned {len(best_alignment)} out of {len(predicted_sentences)} sentences\n")
        return best_alignment, matched_word_indices, updated_pred_sentences
