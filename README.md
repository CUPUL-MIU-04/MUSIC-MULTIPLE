# Music Multiple 🎵
![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

**Music Multiple** is a versatile PyTorch library for music generation, forked from Meta's AudioCraft with enhanced features for multiple genres, durations, and music styles. Music Multiple contains inference and training code for state-of-the-art AI generative models producing high-quality music and audio.

## 🚀 New Features in Music Multiple

- **🎵 Multiple Music Genres**: Enhanced support for Latin genres (salsa, reggaeton, bachata, cumbia) and international styles
- **⏱️ Flexible Durations**: Generate music from short clips (10s) to extended compositions (5min+)
- **🌍 Multi-language Support**: Improved text processing for Spanish and other languages
- **🔄 Extended Integrations**: Compatibility with Magenta and Stable Audio Tools
- **🎛️ Advanced Controls**: Fine-grained control over musical parameters and styles

## Installation

Music Multiple requires Python 3.9, PyTorch 2.1.0. To install Music Multiple, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
# Then proceed to one of the following
python -m pip install -U music-multiple  # stable release (coming soon)
python -m pip install -U git+https://github.com/cupul-miu-04/music-multiple#egg=music-multiple  # bleeding edge
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
python -m pip install -e '.[wm]'  # if you want to train a watermarking model
