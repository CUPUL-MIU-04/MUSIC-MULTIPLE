## MAGNET.md
```markdown
# MAGNeT: Masked Audio Generation using a Single Non-Autoregressive Transformer

Music Multiple provides the code and models for MAGNeT, [Masked Audio Generation using a Single Non-Autoregressive Transformer][arxiv].

MAGNeT is a text-to-music and text-to-sound model capable of generating high-quality audio samples conditioned on text descriptions.
It is a masked generative non-autoregressive Transformer trained over a 32kHz EnCodec tokenizer with 4 codebooks sampled at 50 Hz. 
Unlike prior work on masked generative audio Transformers, such as [SoundStorm](https://arxiv.org/abs/2305.09636) and [VampNet](https://arxiv.org/abs/2307.04686), 
MAGNeT doesn't require semantic token conditioning, model cascading or audio prompting, and employs a full text-to-audio using a single non-autoregressive Transformer.

Check out our [sample page][magnet_samples] or test the available demo!

We use 16K hours of licensed music to train MAGNeT. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MAGNET_MODEL_CARD.md).


## Installation

Please follow the Music Multiple installation instructions from the [README](../README.md).

Music Multiple requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

We currently offer two ways to interact with MAGNeT:
1. You can use the gradio demo locally by running [`python -m demos.magnet_app --share`](../demos/magnet_app.py).
2. You can play with MAGNeT by running the jupyter notebook at [`demos/magnet_demo.ipynb`](../demos/magnet_demo.ipynb) locally (if you have a GPU).

## API

We provide a simple API and 6 pre-trained models. The pre trained models are:
- `facebook/magnet-small-10secs`: 300M model, text to music, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-small-10secs)
- `facebook/magnet-medium-10secs`: 1.5B model, text to music, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-medium-10secs)
- `facebook/magnet-small-30secs`: 300M model, text to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-small-30secs)
- `facebook/magnet-medium-30secs`: 1.5B model, text to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-medium-30secs)
- `facebook/audio-magnet-small`: 300M model, text to sound-effect - [🤗 Hub](https://huggingface.co/facebook/audio-magnet-small)
- `facebook/audio-magnet-medium`: 1.5B model, text to sound-effect - [🤗 Hub](https://huggingface.co/facebook/audio-magnet-medium)

In order to use MAGNeT locally **you must have a GPU**. We recommend 16GB of memory, especially for 
the medium size models. 

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained('facebook/magnet-small-10secs')
descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)