import torch
from models.generator import Generator


def spec2wav(model: Generator, mel: torch.Tensor, f0: torch.Tensor):
    """
    mel: (B, seq_len, num_mel)
    f0: (B, seq_len)
    Example usage:

    ```python
    from audio_commons import get_mel_torch
    from pitch_extraction import get_pitch

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
    ```
    """
    with torch.no_grad():
        c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1)
        y = model(c, f0.unsqueeze(0)).view(-1)
    wav_out = y.cpu().numpy()
    return wav_out
