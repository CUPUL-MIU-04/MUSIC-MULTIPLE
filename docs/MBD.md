# MultiBand Diffusion

Music Multiple provides the code and models for MultiBand Diffusion, [From Discrete Tokens to High Fidelity Audio using MultiBand Diffusion][arxiv].
MultiBand diffusion is a collection of 4 models that can decode tokens from
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> into waveform audio. You can listen to some examples on the <a href="https://ai.honu.io/papers/mbd/">sample page</a>.

<a target="_blank" href="https://colab.research.google.com/drive/1JlTOjB-G0A2Hz3h8PK63vLZk4xdCI5QB?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<br>


## Installation

Please follow the Music Multiple installation instructions from the [README](../README.md).


## Usage

We offer a number of way to use MultiBand Diffusion:
1. The MusicGen demo includes a toggle to try diffusion decoder. You can use the demo locally by running [`python -m demos.musicgen_app --share`](../demos/musicgen_app.py), or through the [MusicGen Colab](https://colab.research.google.com/drive/1JlTOjB-G0A2Hz3h8PK63vLZk4xdCI5QB?usp=sharing).
2. You can play with MusicGen by running the jupyter notebook at [`demos/musicgen_demo.ipynb`](../demos/musicgen_demo.ipynb) locally (if you have a GPU).

## API

We provide a simple API and pre-trained models for MusicGen and for EnCodec at 24 khz for 3 bitrates (1.5 kbps, 3 kbps and 6 kbps).

See after a quick example for using MultiBandDiffusion with the MusicGen API:

```python
import torchaudio
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
mbd = MultiBandDiffusion.get_mbd_musicgen()
model.set_generation_params(duration=8)  # generate 8 seconds.
wav, tokens = model.generate_unconditional(4, return_tokens=True)    # generates 4 unconditional audio samples and keep the tokens for MBD generation
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav_diffusion = mbd.tokens_to_wav(tokens)
wav, tokens = model.generate(descriptions, return_tokens=True)  # generates 3 samples and keep the tokens.
wav_diffusion = mbd.tokens_to_wav(tokens)
melody, sr = torchaudio.load('./assets/bach.mp3')
# Generates using the melody from the given audio and the provided descriptions, returns audio and audio tokens.
wav, tokens = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr, return_tokens=True)
wav_diffusion = mbd.tokens_to_wav(tokens)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav and {idx}_diffusion.wav, with loudness normalization at -14 db LUFS for comparing the methods.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    audio_write(f'{idx}_diffusion', wav_diffusion[idx].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)