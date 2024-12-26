import asyncio
import os

import click
import numpy as np
import pandas as pd
from jiwer import cer, wer
from joblib import Parallel, delayed
from pydub import AudioSegment
from pytriton.client import AsyncioModelClient
from tqdm import tqdm


async def get_texts_from_audio_by_asr(triton_address, triton_port, dataset_dir, input_batch):
    results = {}
    pending_responces = {}

    client = AsyncioModelClient(f"{triton_address}:{triton_port}", "ensemble_english_stt", inference_timeout_s=600)

    async with asyncio.TaskGroup() as tg:
        for input_file in input_batch:
            input_path = os.path.join(dataset_dir, input_file)
            txt_path = input_path.replace("/wavs", "/asr_recognized_texts").replace(".wav", ".txt")

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="UTF-8") as text_file:
                    text = text_file.read()
                results[input_file] = text
            else:
                audio = AudioSegment.from_wav(input_path).set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float64)

                result = tg.create_task(client.infer_sample(audio_signal=audio_data))
                pending_responces[input_file] = result  # .tolist()[0]

    await client.close()

    for input_file, responce in pending_responces.items():
        input_path = os.path.join(dataset_dir, input_file)
        txt_path = input_path.replace("/wavs", "/asr_recognized_texts").replace(".wav", ".txt")

        text = responce.result()["decoded_texts"].decode("UTF-8")

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(text)

        results[input_file] = text

    return results


def process_audios(input_batch, dataset_dir, triton_address, triton_port, tqdm_bar):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    recognized_texts = loop.run_until_complete(
        get_texts_from_audio_by_asr(
            input_batch=input_batch,
            dataset_dir=dataset_dir,
            triton_address=triton_address,
            triton_port=triton_port,
        )
    )

    tqdm_bar.update(len(input_batch))

    return recognized_texts


@click.command()
@click.option("--dataset_path", help="Path to the dataset containing audio files.")
@click.option("--wer_threshold", type=float, default=0.5, help="WER threshold.")
@click.option("--cer_threshold", type=float, default=0.5, help="CER threshold.")
@click.option("--triton_address", default="localhost", help="Address of the Triton Inference Server.")
@click.option("--triton_port", type=int, default=8000, help="Port of the Triton Inference Server.")
@click.option("--batch_size", type=int, default=10, help="Batch size for processing audio files.")
@click.option(
    "--n_jobs", type=int, default=-1, help="Number of parallel jobs to use while processing. -1 means to use all cores."
)
def process_dataset(dataset_path, wer_threshold, cer_threshold, triton_address, triton_port, batch_size, n_jobs):
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path, sep="|")

    assert "path_to_wav" in metadata_df.columns
    assert "text" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    if not os.path.exists(os.path.join(dataset_path, "metadata_before_ASR.csv")):
        metadata_df.to_csv(os.path.join(dataset_path, "metadata_before_ASR.csv"), sep="|", index=False)

    files = metadata_df["path_to_wav"].values
    status_bar = tqdm(files, total=len(files), desc="Processing audio files")

    batches = [files[i : min(i + batch_size, len(files))] for i in range(0, len(files), batch_size)]

    list_of_recognized_texts = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(process_audios)(
            input_batch=batch,
            dataset_dir=dataset_path,
            tqdm_bar=status_bar,
            triton_address=triton_address,
            triton_port=triton_port,
        )
        for batch in batches
    )

    recognized_texts = {path: text for batch in list_of_recognized_texts for path, text in batch.items()}

    metadata_df["recognized_text"] = metadata_df["path_to_wav"].map(recognized_texts)

    recognized_text_empty_mask = metadata_df["recognized_text"].isnull()
    print(
        f"Found {sum(recognized_text_empty_mask)} samples for which ASR did not recognized any speech. Deleting from dataset."
    )
    metadata_df = metadata_df[~recognized_text_empty_mask]

    original_text_empty_mask = metadata_df["text"].isnull()
    print(f"Found {sum(original_text_empty_mask)} samples for which there are no texts found. Deleting from dataset.")
    metadata_df = metadata_df[~original_text_empty_mask]

    metadata_df["wer"] = metadata_df.apply(lambda row: wer(row["text"], row["recognized_text"]), axis=1)
    metadata_df["cer"] = metadata_df.apply(lambda row: cer(row["text"], row["recognized_text"]), axis=1)

    wer_mask = metadata_df["wer"] >= wer_threshold
    print(
        f"Found {sum(wer_mask)} samples for which WER threshold of {wer_threshold:.2f} exceeded. Deleting from dataset."
    )
    metadata_df = metadata_df[~wer_mask]

    cer_mask = metadata_df["cer"] >= cer_threshold
    print(
        f"Found {sum(cer_mask)} samples for which CER threshold of {cer_threshold:.2f} exceeded. Deleting from dataset."
    )
    metadata_df = metadata_df[~cer_mask]

    metadata_df[["path_to_wav", "speaker_id", "text"]].to_csv(
        os.path.join(dataset_path, "metadata.csv"), sep="|", index=False
    )


if __name__ == "__main__":
    process_dataset()
