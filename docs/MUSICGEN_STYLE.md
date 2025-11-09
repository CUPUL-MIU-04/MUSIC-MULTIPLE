# MusicGen-Style: Audio Conditioning for Music Generation via Discrete Bottleneck Features

Music Multiple provides the code and models for MusicGen-Style, [Audio Conditioning for Music Generation via Discrete Bottleneck Features][arxiv].

MusicGen-Style is a text-and-audio-to-music model that can be conditioned on textual and audio data (thanks to a style conditioner). 
The style conditioner takes as input a music excerpt of a few seconds (between 1.5 and 4.5) extracts some features that are used by the model to generate music in the same style. 
This style conditioning can be mixed with textual description. 

Check out our [sample page][musicgen_style_samples] or test the available demo!

We use 16K hours of licensed music to train MusicGen-Style. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MUSICGEN_STYLE_MODEL_CARD.md).


## Installation

Please follow the Music Multiple installation instructions from the [README](../README.md).

MusicGen-Stem requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

1. You can play with MusicGen-Style by running the jupyter notebook at [`demos/musicgen_style_demo.ipynb`](../demos/musicgen_style_demo.ipynb) locally (if you have a GPU).
2. You can use the gradio demo locally by running python -m demos.musicgen_style_app --share.
3. You can play with MusicGen by running the jupyter notebook at demos/musicgen_style_demo.ipynb locally (if you have a GPU).

## API

We provide a simple API 1 pre-trained model with MERT used as a feature extractor for the style conditioner:
- `facebook/musicgen-style`: medium (1.5B) MusicGen model, text and style to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/musicgen-style)

In order to use MusicGen-Style locally **you must have a GPU**. We recommend 16GB of memory. 

See after a quick example for using the API.

To perform text-to-music:
```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-style')


model.set_generation_params(
    duration=8, # generate 8 seconds, can go up to 30
    use_sampling=True, 
    top_k=250,
    cfg_coef=3., # Classifier Free Guidance coefficient 
    cfg_coef_beta=None, # double CFG is only useful for text-and-style conditioning
)  

descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)