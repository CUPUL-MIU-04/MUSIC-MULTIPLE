# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified by Music Multiple
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
import torch
import gradio as gr  # type: ignore
from audiocraft.data.audio_utils import f32_pcm, normalize_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import JASCO
# flake8: noqa

MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
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
        self.files = []  # type: ignore

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


def chords_string_to_list(chords: str):
    if chords == '':
        return []

    # clean white spaces or [ ] chars
    chords = chords.replace('[', '')
    chords = chords.replace(']', '')
    chords = chords.replace(' ', '')
    chrd_times = [x.split(',') for x in chords[1:-1].split('),(')]
    return [(x[0], float(x[1])) for x in chrd_times]


def load_model(version='facebook/jasco-chords-drums-400M'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = None  # in case loading would crash
        MODEL = JASCO.get_pretrained(version)


def _do_predictions(texts, chords, melody_matrix, drum_prompt, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(**gen_kwargs)
    be = time.time()

    # preprocess chords: str to list of tuples
    chords = chords_string_to_list(chords)

    if melody_matrix is not None:
        melody_matrix = torch.load(melody_matrix.name, weights_only=True)
        if len(melody_matrix.shape) != 2:
            raise gr.Error(f"Melody matrix should be a torch tensor of shape [n_melody_bins, T]; got: {melody_matrix.shape}")
        if melody_matrix.shape[0] > melody_matrix.shape[1]:
            melody_matrix = melody_matrix.permute(1, 0)

    # preprocess drums
    if drum_prompt is None:
        preprocessed_drums_wav = None
        drums_sr = 32000
    else:
        # gradio loads audio in int PCM 16-bit, we need to convert it to float32
        drums_sr, drums = drum_prompt[0], f32_pcm(torch.from_numpy(drum_prompt[1])).t()
        if drums.dim() == 1:
            drums = drums[None]

        drums = normalize_audio(drums, strategy="loudness", loudness_headroom_db=16, sample_rate=drums_sr)
        preprocessed_drums_wav = drums
    try:
        outputs = MODEL.generate_music(descriptions=texts, chords=chords,
                                       drums_wav=preprocessed_drums_wav,
                                       melody_salience_matrix=melody_matrix,
                                       drums_sample_rate=drums_sr, progress=progress)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    outputs = outputs.detach().cpu().float()
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_wavs


def predict_full(model,
                 text, chords_sym, melody_file,
                 drums_file, drums_mic, drum_input_src,
                 cfg_coef_all, cfg_coef_txt,
                 ode_rtol, ode_atol,
                 ode_solver, ode_steps,
                 progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    drums = drums_mic if drum_input_src == "mic" else drums_file
    wavs = _do_predictions(
        texts=[text] * 2,  # we generate two audio outputs for each input prompt
        chords=chords_sym,
        drum_prompt=drums,
        melody_matrix=melody_file,
        progress=True,
        gradio_progress=progress,
        cfg_coef_all=cfg_coef_all,
        cfg_coef_txt=cfg_coef_txt,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        euler=ode_solver == 'euler',
        euler_steps=ode_steps)

    return wavs


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Music Multiple - JASCO Advanced Music Generation
            **Music Multiple** presents JASCO - Advanced text-to-music with temporal control over melodies, chords, and beats.

            Based on the research: ["Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation"]
                          (https://arxiv.org/abs/2406.10970)

            *Part of the Music Multiple ecosystem - Enhanced multi-genre music generation platform*
            """
        )
        # Submit | generated
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Generate Music", variant="primary")
                # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)

            with gr.Column():
                audio_output_0 = gr.Audio(label="Generated Audio 1", type='filepath')
                audio_output_1 = gr.Audio(label="Generated Audio 2", type='filepath')

        # TEXT | models
        with gr.Row():
            with gr.Column():
                text = gr.Text(label="Music Description",
                               value="Strings, woodwind, orchestral, symphony.",
                               interactive=True,
                               placeholder="Describe the music you want to generate...")
            with gr.Column():
                model = gr.Radio([
                    'facebook/jasco-chords-drums-400M', 'facebook/jasco-chords-drums-1B',
                    'facebook/jasco-chords-drums-melody-400M', 'facebook/jasco-chords-drums-melody-1B',
                    ],
                 label="Model Version", value='facebook/jasco-chords-drums-melody-400M', interactive=True)

        # CHORDS
        gr.Markdown("🎵 Chord Progression Controls")
        with gr.Row():
            chords_sym = gr.Text(label="Chord Progression",
                                 value="(C, 0.0), (D, 2.0), (F, 4.0), (Ab, 6.0), (Bb, 7.0), (C, 8.0)",
                                 interactive=True,
                                 placeholder="Format: (chord, start_time), (chord, start_time)...")

        # DRUMS
        gr.Markdown("🥁 Drum Controls")
        with gr.Row():
            drum_input_src = gr.Radio(["file", "mic"], value="file",
                                      label="Drum Input Source")
            drums_file = gr.Audio(sources=["upload"], type="numpy", label="Upload Drum Track",
                                  interactive=True, elem_id="drums-input")

            drums_mic = gr.Audio(sources=["microphone"], type="numpy", label="Record Drums",
                                 interactive=True, elem_id="drums-mic-input")

        # MELODY
        gr.Markdown("🎶 Melody Controls")
        with gr.Row():
            melody_file = gr.File(label="Upload Melody Matrix", interactive=True, elem_id="melody-file-input")

        # CFG params
        gr.Markdown("⚙️ Advanced Generation Parameters")
        with gr.Row():
            cfg_coef_all = gr.Number(label="CFG All", value=1.25, step=0.25, interactive=True)
            cfg_coef_txt = gr.Number(label="CFG Text", value=2.5, step=0.25, interactive=True)
            ode_tol = gr.Number(label="ODE Solver Tolerance", 
                                value=1e-4, step=1e-5, interactive=True)
            ode_solver = gr.Radio(['euler', 'dopri5'], label="ODE Solver", value='euler', interactive=True)
            ode_steps = gr.Number(label="Euler Steps", value=10, step=1, interactive=True)

        submit.click(fn=predict_full,
                     inputs=[model,
                             text, chords_sym, melody_file,
                             drums_file, drums_mic, drum_input_src,
                             cfg_coef_all, cfg_coef_txt, ode_tol, ode_tol, ode_solver, ode_steps],
                     outputs=[audio_output_0, audio_output_1])
        
        gr.Markdown("### 🎧 Example Prompts")
        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "80s pop with groovy synth bass and electric piano",
                    "(N, 0.0), (C, 0.32), (Dm7, 3.456), (Am, 4.608), (F, 8.32), (C, 9.216)",
                    "./assets/salience_2.th",
                    "./assets/salience_2.wav",
                ],
                [
                    "Strings, woodwind, orchestral, symphony.",                         # text
                    "(C, 0.0), (D, 2.0), (F, 4.0), (Ab, 6.0), (Bb, 7.0), (C, 8.0)",     # chords
                    None,                                                               # melody
                    None,                                                               # drums
                ],
                [
                    "distortion guitars, heavy rock, catchy beat",
                    "",
                    None,
                    "./assets/sep_drums_1.mp3",
                ],
                [
                    "hip hop beat with a catchy melody and a groovy bass line",
                    "",
                    None,
                    "./assets/CJ_Beatbox_Loop_05_90.wav",
                ],
                [
                    "hip hop beat with a catchy melody and a groovy bass line",
                    "(C, 0.0), (D, 2.0), (F, 4.0), (Ab, 6.0), (Bb, 7.0), (C, 8.0)",
                    None,
                    "./assets/CJ_Beatbox_Loop_05_90.wav",
                ],
            ],
            inputs=[text, chords_sym, melody_file, drums_file],
            outputs=[audio_output_0, audio_output_1]
        )
        
        gr.Markdown(
            """
            ### 🎵 About Music Multiple - JASCO

            **Music Multiple** brings you JASCO - Advanced music generation with precise temporal controls.
            Generate 10-second music clips based on text descriptions with fine-grained control over:
            - **Chords**: Specify exact chord progressions and timing
            - **Drums**: Upload or record drum patterns
            - **Melody**: Upload melody matrices for precise melodic control

            ### Model Variants Available:
            1. **jasco-chords-drums-400M** - 400M parameters, text + chords + drums
            2. **jasco-chords-drums-1B** - 1B parameters, text + chords + drums  
            3. **jasco-chords-drums-melody-400M** - 400M parameters, text + chords + drums + melody
            4. **jasco-chords-drums-melody-1B** - 1B parameters, text + chords + drums + melody

            *Part of the Music Multiple ecosystem - Enhanced audio generation platform*
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