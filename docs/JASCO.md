## JASCO.md
```markdown
# JASCO: Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation

Music Multiple provides the code and models for JASCO, [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation][arxiv].

We present JASCO, a temporally controlled text-to-music generation model utilizing both symbolic and audio-based conditions.
JASCO can generate high-quality music samples conditioned on global text descriptions along with fine-grained local controls.
JASCO is based on the Flow Matching modeling paradigm together with a novel conditioning method, allowing for music generation controlled both locally (e.g., chords) and globally (text description).

Check out our [sample page][sample_page] or test the available demo!

We use ~16K hours of licensed music to train JASCO. 


## Model Card

See [the model card](../model_cards/JASCO_MODEL_CARD.md).


## Installation

First, Please follow the Music Multiple installation instructions from the [README](../README.md).

Then, download and install chord_extractor from [source](http://www.isophonics.net/nnls-chroma)

See further required installation under **Data Preprocessing** section

## Usage

We currently offer two ways to interact with JASCO:
1. You can use the gradio demo locally by running [`python -m demos.jasco_app`](../demos/jasco_app.py), you can add `--share` to deploy a sharable space mounted on your device.
2. You can play with JASCO by running the jupyter notebook at [`demos/jasco_demo.ipynb`](../demos/jasco_demo.ipynb) locally.

## API

We provide a simple API and pre-trained models:
- `facebook/jasco-chords-drums-400M`: 400M model, text to music with chords and drums support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-400M)
- `facebook/jasco-chords-drums-1B`: 1B model, text to music with chords and drums support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-1B)
- `facebook/jasco-chords-drums-melody-400M`: 400M model, text to music with chords, drums and melody support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-melody-400M)
- `facebook/jasco-chords-drums-melody-1B`: 1B model, text to music with chords, drums and melody support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-melody-1B)


See after a quick example for using the API.

```python
from audiocraft.models import JASCO

model = JASCO.get_pretrained('facebook/jasco-chords-drums-400M', chords_mapping_path='../assets/chord_to_index_mapping.pkl')

model.set_generation_params(
    cfg_coef_all=5.0,
    cfg_coef_txt=0.0
)

# set textual prompt
text = "Strings, woodwind, orchestral, symphony."

# define chord progression
chords = [('C', 0.0), ('D', 2.0), ('F', 4.0), ('Ab', 6.0), ('Bb', 7.0), ('C', 8.0)]

# run inference
output = model.generate_music(descriptions=[text], chords=chords, progress=True)

audio_write('output', output.cpu().squeeze(0), model.sample_rate, strategy="loudness", loudness_compressor=True)