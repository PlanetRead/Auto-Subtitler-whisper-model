"""
This file contains the schemas for the data classes used in the consolidation.

All rights reserved.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class WordTimestamp:
    """
    Data class to store word and its timestamps.
    """
    word: str
    start: float
    end: float

@dataclass
class TimeAlignedSentence:
    """
    Data class to store a sentence and its aligned words with timestamps.
    """
    sentence: str
    start_time: float
    end_time: float
    words: List[WordTimestamp]

@dataclass
class AlignmentResult:
    """
    Data class to store the result of alignment between ground truth and predicted sentences.
    """
    ground_truth_sentence: List[str]
    predicted_sentence: TimeAlignedSentence
    similarity_score: float