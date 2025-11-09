# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified by Music Multiple
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..musicgen._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=32, partition=partitions)
    launcher.bind_(solver='musicgen/musicgen_base_32khz')
    launcher.bind_(dset='music_multiple/latin_music_32khz')

    # Configuraciones específicas para música latina
    latin_config = {
        'conditioners.description.t5.word_dropout': 0.1,
        'classifier_free_guidance.training_dropout': 0.1,
        'generate.lm.temperature': 0.8,  # Más creatividad para ritmos latinos
    }

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}

    launcher.bind_(fsdp)

    with launcher.job_array():
        launcher(latin_config)
        launcher({**latin_config, **medium})