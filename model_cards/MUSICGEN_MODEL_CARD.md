# MusicGen Model Card - Music Multiple

## Model details

**Organization developing the model:** The FAIR team of Meta AI.  
**Modified and distributed by:** Music Multiple

**Model date:** MusicGen was trained between April 2023 and May 2023.

**Music Multiple Version:** This is the core model powering Music Multiple, with significant interface enhancements and additional features.

[... resto del contenido técnico igual ...]

## Music Multiple Enhancements

🎵 **Multiple Generation Features:**
- ✅ Multiple duration options (short, medium, long, extended)
- ✅ Latin music genre optimizations
- ✅ Spanish text processing improvements
- ✅ Enhanced melody conditioning interface
- ✅ Batch generation capabilities
- ✅ Multiple audio format exports

🌎 **Language Support:**
- ✅ Improved Spanish prompt handling
- ✅ Localized interface options
- ✅ Multi-language text preprocessing

⚡ **Performance Improvements:**
- ✅ Optimized generation parameters
- ✅ Memory usage optimizations
- ✅ Faster inference pipelines

## Usage Example with Music Multiple

```python
from music_multiple import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-small")

# Music Multiple provides enhanced generation options
audio = model.generate_with_options(
    descriptions=["happy electronic music with latin rhythms"],
    duration="medium",  # Music Multiple feature
    genre="latin",      # Music Multiple feature
    quality="high"      # Music Multiple feature
)