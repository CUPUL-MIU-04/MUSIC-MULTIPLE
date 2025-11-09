# EnCodec: High Fidelity Neural Audio Compression

Music Multiple provides the training code for EnCodec, a state-of-the-art deep learning
based audio codec supporting both mono and stereo audio, presented in the
[High Fidelity Neural Audio Compression][arxiv] paper.
Check out our [sample page][encodec_samples].

## Original EnCodec models

The EnCodec models presented in High Fidelity Neural Audio Compression can be accessed
and used with the [EnCodec repository](https://github.com/facebookresearch/encodec).

**Note**: We do not guarantee compatibility between the Music Multiple and EnCodec codebases
and released checkpoints at this stage.


## Installation

Please follow the Music Multiple installation instructions from the [README](../README.md).


## Training

The [CompressionSolver](../audiocraft/solvers/compression.py) implements the audio reconstruction
task to train an EnCodec model. Specifically, it trains an encoder-decoder with a quantization
bottleneck - a SEANet encoder-decoder with Residual Vector Quantization bottleneck for EnCodec -
using a combination of objective and perceptual losses in the forms of discriminators.

The default configuration matches a causal EnCodec training at a single bandwidth.

### Example configuration and grids

We provide sample configuration and grids for training EnCodec models.

The compression configuration are defined in
[config/solver/compression](../config/solver/compression).

The example grids are available at
[audiocraft/grids/compression](../audiocraft/grids/compression).

```shell
# base causal encodec on monophonic audio sampled at 24 khz
dora grid compression.encodec_base_24khz
# encodec model used for MusicGen on monophonic audio sampled at 32 khz
dora grid compression.encodec_musicgen_32khz