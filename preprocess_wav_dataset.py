# run this on the TSE dataset to clean up files too long or too short

from glob import glob
import os
import torchaudio
import click


def get_audio_length(audio_path: str):
    audio, fs = torchaudio.load(audio_path)
    return audio.shape[-1] / fs


def audio_len_filter(folder: str):
    for spk in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, spk)):
            for path in glob(os.path.join(folder, spk, "*.wav")):
                if not 3 <= get_audio_length(path) <= 15:
                    os.remove(path)


@click.command()
@click.argument(
    "folder", nargs=1, type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def main(folder: str):
    audio_len_filter(folder)


if __name__ == "__main__":
    main()
