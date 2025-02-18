"""
This file contains the code for the inference of the openai whisper model

All rights reserved.
"""

import os
import sys

import whisper
from consolidate.logger import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class WhisperInference:

    def __init__(self, checkpoint_path: str = "/content/whispermodel.pt", language: str = "pa", initial_prompt: str = "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਇੱਕ ਪੰਜਾਬੀ ਆਡੀਓ ਫਾਇਲ ਹੈ, ਇਸ ਦਾ ਸਮੱਗਰੀ ਹੇਠ ਲਿਖੀ ਹੈ।"):
        """
        Initialize the Whisper model for inference.
        """
        self.checkpoint_path = checkpoint_path
        self.language = language
        self.initial_prompt = initial_prompt
        self.load_model()

    def load_model(self):
        """
        A method to load the Whisper model.
        """
        self.model = whisper.load_model(self.checkpoint_path, device="cuda")
        logger.info(f"Model loaded successfully from checkpoint: {self.checkpoint_path}")

    def run_inference(self, audio_path: str):
        """
        Run inference on the provided audio file.
        """
        result = self.model.transcribe(audio_path, language=self.language, word_timestamps=True, initial_prompt=self.initial_prompt)
        return result