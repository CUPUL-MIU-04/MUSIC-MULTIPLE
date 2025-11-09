## MUSICGEN.md
```markdown
# MusicGen: Simple and Controllable Music Generation

Music Multiple provides the code and models for MusicGen, [a simple and controllable model for music generation][arxiv].
MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz.
Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require
a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing
a small delay between the codebooks, we show we can predict them in parallel, thus having only 50 auto-regressive
steps per second of audio.
Check out our [sample page][musicgen_samples] or test the available demo!

<a target="_blank" href="https://ai.honu.io/red/musicgen-colab">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/facebook/MusicGen">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>
<br>

We use 20K hours of licensed music to train MusicGen. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MUSICGEN_MODEL_CARD.md).


## Installation

Please follow the Music Multiple installation instructions from the [README](../README.md).

Music Multiple requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

We offer a number of way to interact with MusicGen:
1. A demo is also available on the [`facebook/MusicGen` Hugging Face Space](https://huggingface.co/spaces/facebook/MusicGen)
(huge thanks to all the HF team for their support).
2. You can run the extended demo on a Colab:
[colab notebook](https://ai.honu.io/red/musicgen-colab)
3. You can use the gradio demo locally by running [`python -m demos.musicgen_app --share`](../demos/musicgen_app.py).
4. You can play with MusicGen by running the jupyter notebook at [`demos/musicgen_demo.ipynb`](../demos/musicgen_demo.ipynb) locally (if you have a GPU).
5. Finally, checkout [@camenduru Colab page](https://github.com/camenduru/MusicGen-colab)
which is regularly updated with contributions from @camenduru and the community.


## API

We provide a simple API and 10 pre-trained models. The pre trained models are:
- `facebook/musicgen-small`: 300M model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-small)
- `facebook/musicgen-medium`: 1.5B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-medium)
- `facebook/musicgen-melody`: 1.5B model, text to music and text+melody to music - [🤗 Hub](https://huggingface.co/facebook/musicgen-melody)
- `facebook/musicgen-large`: 3.3B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-large)
- `facebook/musicgen-melody-large`: 3.3B model, text to music and text+melody to music - [🤗 Hub](https://huggingface.co/facebook/musicgen-melody-large)
- `facebook/musicgen-stereo-*`: All the previous models fine tuned for stereo generation -
    [small](https://huggingface.co/facebook/musicgen-stereo-small),
    [medium](https://huggingface.co/facebook/musicgen-stereo-medium),
    [large](https://huggingface.co/facebook/musicgen-stereo-large),
    [melody](https://huggingface.co/facebook/musicgen-stereo-melody),
    [melody large](https://huggingface.co/facebook/musicgen-stereo-melody-large).

We observe the best trade-off between quality and compute with the `facebook/musicgen-medium` or `facebook/musicgen-melody` model.
In order to use MusicGen locally **you must have a GPU**. We recommend 16GB of memory, but smaller
GPUs will be able to generate short sequences, or longer sequences with the `facebook/musicgen-small` model.

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)