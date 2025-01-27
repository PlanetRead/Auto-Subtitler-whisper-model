# Subtitle Alignment

A tool for automatic subtitle alignment using Whisper model fine-tuning and similarity-based matching.

## Overview

This project provides a solution for aligning subtitles with audio/video content using OpenAI's Whisper model. It combines speech recognition capabilities with similarity scoring to ensure accurate subtitle timing and placement.

## Features

- Whisper model fine-tuning for improved speech recognition
- Similarity-based subtitle alignment
- Subtitle Error Rate (SER) measurement
- Automated alignment correction

## How It Works

1. **Speech Recognition**: Uses a fine-tuned Whisper model to transcribe audio content
2. **Similarity Matching**: Compares transcribed text with existing subtitles using similarity scores
3. **Alignment**: Adjusts subtitle timing based on matched segments
4. **Quality Assessment**: Measures Subtitle Error Rate to evaluate alignment accuracy

## Installation

```bash
git clone https://github.com/PlanetRead/Auto-Subtitler-whisper-model.git/
mv Auto-Subtitler-whisper-model subtitle-alignment
cd subtitle-alignment

conda create -n alignment python=3.10.16
conda activate alignment
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```


# Consolidate Pipeline

This pipeline will be used to inference the whisper model and if user have .srt and text file of the ground truth,
this pipeline can be used to calculate the error rate.

```bash
python consolidate\main.py --audio_file <audio_file_path> [--srt_file <srt_file_path>] [--text_file <text_file_path>]
```

SRT file and Text files are optional, If the SRT file is provided then the text file must be provided.
