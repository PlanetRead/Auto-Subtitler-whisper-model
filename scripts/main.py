from utils import split_audio_and_srt_for_finetuning

if __name__ == "__main__":
    # input_dir = "data/Punjabi/Ground Truth SRT"
    # output_dir = "data/Punjabi/Text"

    # convert_srt_directory(input_dir, output_dir)

    audio_dir = "data/Punjabi/Audio"
    srt_dir = "data/Punjabi/Ground Truth SRT"
    output_dir = "data/Punjabi/Finetuning"

    split_audio_and_srt_for_finetuning(audio_dir, srt_dir, output_dir)
