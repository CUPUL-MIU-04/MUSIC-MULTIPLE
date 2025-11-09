# AudioGen: Textually-guided audio generation

Music Multiple provides the code and a model re-implementing AudioGen, a [textually-guided audio generation][audiogen_arxiv]
model that performs text-to-sound generation.

The provided AudioGen reimplementation follows the LM model architecture introduced in [MusicGen][musicgen_arxiv]
and is a single stage auto-regressive Transformer model trained over a 16kHz
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz.
This model variant reaches similar audio quality than the original implementation introduced in the AudioGen publication
while providing faster generation speed given the smaller frame rate.

**Important note:** The provided models are NOT the original models used to report numbers in the
[AudioGen publication][audiogen_arxiv]. Refer to the model card to learn more about architectural changes.

Listen to samples from the **original AudioGen implementation** in our [sample page][audiogen_samples].


## Model Card

See [the model card](../model_cards/AUDIOGEN_MODEL_CARD.md).


## Installation

Please follow the Music Multiple installation instructions from the [README](../README.md).

Music Multiple requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## API and usage

We provide a simple API and 1 pre-trained models for AudioGen:

`facebook/audiogen-medium`: 1.5B model, text to sound - [🤗 Hub](https://huggingface.co/facebook/audiogen-medium)

You can play with AudioGen by running the jupyter notebook at [`demos/audiogen_demo.ipynb`](../demos/audiogen_demo.ipynb) locally (if you have a GPU).

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # generate 5 seconds.
descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)