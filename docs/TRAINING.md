## TRAINING.md
```markdown
# Music Multiple training pipelines

Music Multiple training pipelines are built on top of PyTorch as our core deep learning library
and [Flashy](https://github.com/facebookresearch/flashy) as our training pipeline design library,
and [Dora](https://github.com/facebookresearch/dora) as our experiment manager.
Music Multiple training pipelines are designed to be research and experiment-friendly.


## Environment setup

For the base installation, follow the instructions from the [README.md](../README.md).
Below are some additional instructions for setting up the environment to train new models.

### Team and cluster configuration

In order to support multiple teams and clusters, Music Multiple uses an environment configuration.
The team configuration allows to specify cluster-specific configurations (e.g. SLURM configuration),
or convenient mapping of paths between the supported environments.

Each team can have a yaml file under the [configuration folder](../config). To select a team set the
`AUDIOCRAFT_TEAM` environment variable to a valid team name (e.g. `labs` or `default`):
```shell
conda env config vars set AUDIOCRAFT_TEAM=default