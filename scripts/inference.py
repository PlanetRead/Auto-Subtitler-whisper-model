"""
This script is used to run inference on the whisper model.

All rights reserved.
"""

import os
import whisper
from copy import deepcopy
from pathlib import Path
import soundfile as sf
import logging
import pandas as pd
from typing import List
import json

logging.basicConfig(level=logging.INFO)

class WhisperInference:
    def __init__(self, model_name: str = "large-v2"):
        self.data_path = Path("data/Punjabi")
        self.audio_path = self.data_path / "Audio"
        self.transcript_path = self.data_path / "Text"
        self.results_path = self.data_path / "Results"
        self.benchmark_path = self.data_path / "benchmark_list.csv"
        self.model_name = model_name
        self.get_benchmark_filenames()
        self.get_dataset_dict()
        self.benchmark_audio_paths = self.dataset_dict["val"]["audio_path"]
        logging.info(f"Loaded {len(self.benchmark_audio_paths)} benchmark audio paths: {self.benchmark_audio_paths}")

    def load_audio(self, audio_path: str):
        audio = whisper.load_audio(audio_path)
        return audio

    def load_model(self):
        self.model = whisper.load_model(self.model_name)

    def get_benchmark_filenames(self):
        logging.info("Loading benchmark filenames...")
        benchmark_df = pd.read_csv(self.benchmark_path, index_col=0).reset_index(drop=True)
        self.benchmark_filenames = benchmark_df["Story Name"].unique().tolist()
        logging.info(f"Loaded the following {len(self.benchmark_filenames)} benchmark filenames: {self.benchmark_filenames}")

    def get_dataset_dict(self):
        logging.info("Creating dataset dictionary...")
        base_lists = {"audio_path": [], "transcript_path": [], "transcript": [], "sampling_rate": [], "array": []}
        self.dataset_dict = {"train": deepcopy(base_lists), "val": deepcopy(base_lists)}

        for dir_path in self.audio_path.glob("*"):
            for file_path in dir_path.glob("*.wav"):
                if any(benchmark_filename in file_path.name for benchmark_filename in self.benchmark_filenames):
                    dataset_type = "val"
                else:
                    dataset_type = "train"

                # Append audio path and transcript path to dataset
                file_audio_path = str(file_path)
                file_transcript_path = str(self.transcript_path / dir_path.name.replace("Videos", "Text") / str(file_path.name.replace(".wav", ".txt")))
                self.dataset_dict[dataset_type]["audio_path"].append(file_audio_path)
                self.dataset_dict[dataset_type]["transcript_path"].append(file_transcript_path)


    def run_inference(self, audio_path: str):
        initial_prompt = "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਇੱਕ ਪੰਜਾਬੀ ਆਡੀਓ ਫਾਇਲ ਹੈ, ਇਸ ਦਾ ਸਮੱਗਰੀ ਹੇਠ ਲਿਖੀ ਹੈ।"
        result = self.model.transcribe(audio_path, language="pa", word_timestamps=True, initial_prompt=initial_prompt)
        return result

    def run_batch_inference(self, audio_paths: List[str]):
        self.load_model()
        logging.info("Running batch inference...")
        for file_audio_path in audio_paths:
            logging.info(f"Running inference on {file_audio_path}")
            result = self.run_inference(file_audio_path)

            # write results to a json file
            json_path = self.results_path / Path(file_audio_path).name.replace(".wav", ".json")
            os.makedirs(json_path.parent, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(result, f)

if __name__ == "__main__":
    print("Running inference on the model...")
    inference = WhisperInference("large-v2")
    inference.run_batch_inference(inference.benchmark_audio_paths)



