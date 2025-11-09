# Music-Multiple/audiocraft/models/music_multiple.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified by Music Multiple
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Music Multiple - Enhanced version of MusicGen with multiple features:
- Multiple duration presets
- Latin genre support  
- Spanish text processing
- Easy-to-use interface
"""

import typing as tp
import torch
from .musicgen import MusicGen
from ..modules.conditioners import ConditioningAttributes


class MusicMultiple(MusicGen):
    """Music Multiple: Enhanced music generation with multiple features.
    
    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce.
    """
    
    def __init__(self, name: str, compression_model, lm, max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.setup_multiple_features()
        self.set_generation_params(duration=30)  # default duration

    def setup_multiple_features(self):
        """Initialize Music Multiple specific features."""
        # Multiple duration system
        self.available_durations = {
            'short': 15,      # 15 seconds - for social media
            'medium': 30,     # 30 seconds - standard
            'long': 60,       # 1 minute - extended
            'extended': 120   # 2 minutes - full version
        }
        
        # Multiple genres system (with focus on Latin genres)
        self.available_genres = {
            'latin': {
                'salsa': "latin salsa music with piano, trumpet and congas",
                'reggaeton': "reggaeton beat with dem bow rhythm and synth bass",
                'bachata': "romantic bachata with acoustic guitar and bongos",
                'cumbia': "folk cumbia with accordion and guiro",
                'merengue': "fast merengue with saxophone and tambora"
            },
            'electronic': {
                'house': "four-on-the-floor house beat with synth chords",
                'techno': "driving techno with repetitive synth bass",
                'dubstep': "wobble bass dubstep with heavy drops"
            },
            'world': {
                'afrobeat': "polyrhythmic afrobeat with percussion",
                'reggae': "one drop reggae rhythm with organ skanks"
            }
        }
        
        # Spanish text processing enhancements
        self.spanish_enhancements = {
            'alegre': "happy upbeat major key",
            'triste': "sad emotional minor key", 
            'romantico': "romantic slow tempo",
            'energico': "energetic fast tempo"
        }

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-small', device=None):
        """Return pretrained model adapted for Music Multiple."""
        if device is None:
            device = 'cuda' if torch.cuda.device_count() else 'cpu'
            
        # Load base model
        base_model = MusicGen.get_pretrained(name, device)
        
        # Convert to MusicMultiple
        music_multiple = MusicMultiple(
            name=base_model.name,
            compression_model=base_model.compression_model,
            lm=base_model.lm,
            max_duration=base_model.max_duration
        )
        
        # Copy generation parameters
        music_multiple.generation_params = base_model.generation_params
        music_multiple.duration = base_model.duration
        music_multiple.extend_stride = base_model.extend_stride
        
        return music_multiple

    def set_duration(self, duration_type: str = 'medium'):
        """Set duration using preset types.
        
        Args:
            duration_type (str): Type of duration: 'short', 'medium', 'long', 'extended'
        """
        if duration_type in self.available_durations:
            self.duration = self.available_durations[duration_type]
            print(f"✅ Duration set to {duration_type}: {self.duration} seconds")
        else:
            available = list(self.available_durations.keys())
            raise ValueError(f"Duration type '{duration_type}' not available. Choose from: {available}")

    def enhance_spanish_text(self, text: str) -> str:
        """Enhance Spanish text with musical context."""
        enhanced = text.lower()
        for spanish_word, musical_context in self.spanish_enhancements.items():
            if spanish_word in enhanced:
                enhanced = f"{enhanced} {musical_context}"
        return enhanced

    def generate_with_genre(self, descriptions: tp.List[str], genre: str, 
                           subgenre: tp.Optional[str] = None, progress: bool = False):
        """Generate music with specific genre and optional subgenre.
        
        Args:
            descriptions (list of str): Text descriptions.
            genre (str): Main genre ('latin', 'electronic', 'world').
            subgenre (str, optional): Specific subgenre.
            progress (bool): Show generation progress.
        """
        # Validate genre
        if genre not in self.available_genres:
            available_genres = list(self.available_genres.keys())
            raise ValueError(f"Genre '{genre}' not available. Choose from: {available_genres}")
        
        # Validate subgenre
        if subgenre and subgenre not in self.available_genres[genre]:
            available_subgenres = list(self.available_genres[genre].keys())
            raise ValueError(f"Subgenre '{subgenre}' not available in {genre}. Choose from: {available_subgenres}")
        
        # Enhance descriptions with genre context
        enhanced_descriptions = []
        for desc in descriptions:
            # Enhance Spanish text
            desc_enhanced = self.enhance_spanish_text(desc)
            
            # Add genre context
            if subgenre:
                genre_context = self.available_genres[genre][subgenre]
            else:
                # Use first subgenre as default
                first_subgenre = list(self.available_genres[genre].keys())[0]
                genre_context = self.available_genres[genre][first_subgenre]
            
            final_description = f"{desc_enhanced} {genre_context}"
            enhanced_descriptions.append(final_description)
        
        print(f"🎵 Generating {genre}" + (f" - {subgenre}" if subgenre else ""))
        return self.generate(enhanced_descriptions, progress=progress)

    def list_available_genres(self):
        """List all available genres and subgenres."""
        print("🎵 Available Genres in Music Multiple:")
        for genre, subgenres in self.available_genres.items():
            print(f"  {genre.capitalize()}: {', '.join(subgenres.keys())}")

    def list_available_durations(self):
        """List all available duration presets."""
        print("⏱️ Available Durations in Music Multiple:")
        for duration_type, seconds in self.available_durations.items():
            print(f"  {duration_type}: {seconds} seconds")