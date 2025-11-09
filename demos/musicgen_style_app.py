# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified by Music Multiple
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

from einops import rearrange
import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen, MultiBandDiffusion


MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
                
file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/musicgen-style'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(version)


def load_diffusion():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, melodies, duration, top_k, top_p, temperature, cfg_coef, cfg_coef_beta, eval_q, excerpt_length, progress=False, gradio_progress=None):
    MODEL.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature, cfg_coef=cfg_coef, cfg_coef_beta=cfg_coef_beta)
    MODEL.set_style_conditioner_params(eval_q=eval_q, excerpt_length=excerpt_length)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    try:
        if any(m is not None for m in processed_melodies):
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=USE_DIFFUSION
            )
        else:
            outputs = MODEL.generate(texts, progress=progress, return_tokens=USE_DIFFUSION)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    if USE_DIFFUSION:
        if gradio_progress is not None:
            gradio_progress(1, desc='Running MultiBandDiffusion...')
        tokens = outputs[1]
        outputs_diffusion = MBD.tokens_to_wav(tokens)
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs


def predict_full(model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, double_cfg, cfg_coef_beta, eval_q, excerpt_length, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")
    if eval_q < 1 or eval_q > 6:
        raise gr.Error("eval_q must be an integer between 1 and 6 included.")
    if excerpt_length > 4.5:
        raise gr.Error("excerpt_length must be <= 4.5 seconds")

    topk = int(topk)
    eval_q = int(eval_q)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        progress(0, desc="Loading diffusion model...")
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    if double_cfg != "Yes":
        cfg_coef_beta = None
    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, wavs = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        cfg_coef_beta=cfg_coef_beta, eval_q=eval_q, excerpt_length=excerpt_length,
        gradio_progress=progress)
    if USE_DIFFUSION:
        return videos[0], wavs[0], videos[1], wavs[1]
    return videos[0], wavs[0], None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Music Multiple - Style Conditioning
            **Music Multiple** presents advanced style-conditioned music generation with precise audio control.

            Based on the research: ["Audio Conditioning for Music Generation via Discrete Bottleneck Features"](https://arxiv.org/abs/2407.12563)

            *Part of the Music Multiple ecosystem - Advanced style-controlled music generation*
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Music Description", interactive=True,
                                  placeholder="Describe the music style you want to generate...")
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Style Audio Input Source")
                        melody = gr.Audio(sources=["upload"], type="numpy", label="Upload Style Audio",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Generate Music", variant="primary")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["facebook/musicgen-style"],
                                     label="Model Version", value="facebook/musicgen-style", interactive=True)
                    model_path = gr.Text(label="Custom Model Path (advanced)")
                with gr.Row():
                    decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                       label="Decoder Type", value="Default", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration (seconds)", interactive=True)
                    eval_q = gr.Slider(minimum=1, maximum=6, value=3, step=1, label="RVQ Layers for Style", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="CFG Alpha", value=3.0, interactive=True)
                    double_cfg = gr.Radio(["Yes", "No"], 
                                          label="Double CFG", 
                                          value="Yes", 
                                          interactive=True,
                                          info="Use for both text and audio inputs")
                    cfg_coef_beta = gr.Number(label="CFG Beta", value=5.0, interactive=True)
                    excerpt_length = gr.Number(label="Style Audio Length (≤4.5s)", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music Video")
                audio_output = gr.Audio(label="Generated Audio (WAV)", type='filepath')
                diffusion_output = gr.Video(label="MultiBand Diffusion Output")
                audio_diffusion = gr.Audio(label="MultiBand Diffusion Audio (WAV)", type='filepath')
        submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                     show_progress=False).then(predict_full, inputs=[model, model_path, decoder, text, melody, duration, topk, topp,
                                                                     temperature, cfg_coef, double_cfg, cfg_coef_beta, eval_q, excerpt_length],
                                               outputs=[output, audio_output, diffusion_output, audio_diffusion])
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)

        gr.Markdown("### 🎧 Example Prompts")
        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "80s New Wave with synthesizer",
                    "./assets/electronic.mp3",
                    "facebook/musicgen-style",
                    "Default"
                ],
            ],
            inputs=[text, melody, model, decoder],
            outputs=[output]
        )
        
        gr.Markdown(
            """
            ### 🎵 About Music Multiple - Style Conditioning

            **Music Multiple** provides advanced style-conditioned music generation with three input modes:

            #### 🎯 Generation Modes
            1. **Text Only**: Generate music from descriptions only
               - Recommended: Single CFG with coefficient = 3.0

            2. **Audio Style Only**: Generate music matching a style audio excerpt
               - Audio should be ≤4.5 seconds
               - Recommended length: 1.5-4.5 seconds
               - Recommended: Single CFG with coefficient = 3.0

            3. **Text + Audio**: Combine text descriptions with style audio
               - Recommended: Double CFG with Alpha=3, Beta=4
               - Adjust Beta: Lower for more text adherence, higher for more style adherence

            #### ⚙️ Advanced Controls
            - **RVQ Layers**: Number of residual vector quantization layers for style encoding (1-6)
            - **Double CFG**: Separate guidance for text and style conditioning
            - **Excerpt Length**: Duration of style audio used for conditioning
            - **MultiBand Diffusion**: Enhanced audio quality decoding option

            #### 💡 Usage Tips
            - Use clear, descriptive text for best results
            - Style audio works best with distinctive musical characteristics
            - Experiment with Double CFG parameters for balanced text/style control
            - Try different RVQ layers for varying style extraction intensity

            *Part of the Music Multiple ecosystem - Professional music generation platform*
            """
        )

        interface.queue().launch(**launch_kwargs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    ui_full(launch_kwargs)