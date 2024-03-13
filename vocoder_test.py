from models.generator import Generator, GeneratorArgs
from models.vocoder import spec2wav
import soundfile
import json
import torchaudio
import torchcrepe
import torch
import numpy as np
from audio_commons import get_mel_torch
from pitch_extraction import get_pitch


if __name__ == "__main__":
    import sys

    with open(
        "weights/nsf_hifigan_44.1k_hop512_128bin_2024.02/config.json",
        "r",
        encoding="utf-8",
    ) as f:
        model = Generator(GeneratorArgs.from_json(json.load(f)))
    state_dict = torch.load(
        "weights/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt"
    )["generator"]
    model.load_state_dict(state_dict)

    y, sr = torchaudio.load(sys.argv[1])
    y = torchaudio.functional.resample(y, sr, 44100)
    mel = get_mel_torch(y.flatten(), 44100)

    pitch = get_pitch(y.flatten(), mel, 512)
    soundfile.write(
        "test.wav",
        spec2wav(
            model,
            torch.from_numpy(mel),
            torch.from_numpy(pitch),
        ),
        44100,
        format="WAV",
    )
