## METRICS.md
```markdown
# Music Multiple objective metrics

In addition to training losses, Music Multiple provides a set of objective metrics
for audio synthesis and audio generation. As these metrics may require
extra dependencies and can be costly to train, they are often disabled by default.
This section provides guidance for setting up and using these metrics in
the Music Multiple training pipelines.

## Available metrics

### Audio synthesis quality metrics

#### SI-SNR

We provide an implementation of the Scale-Invariant Signal-to-Noise Ratio in PyTorch.
No specific requirement is needed for this metric. Please activate the metric at the
evaluation stage with the appropriate flag:

**Warning:** We report the opposite of the SI-SNR, e.g. multiplied by -1. This is due to internal 
    details where the SI-SNR score can also be used as a training loss function, where lower
    values should indicate better reconstruction. Negative values are such expected and a good sign! Those should be again multiplied by `-1` before publication :)

```shell
dora run <...> evaluate.metrics.sisnr=true