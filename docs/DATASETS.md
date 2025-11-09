# Music Multiple datasets

Our dataset manifest files consist in 1-json-per-line files, potentially gzipped,
as `data.jsons` or `data.jsons.gz` files. This JSON contains the path to the audio
file and associated metadata. The manifest files are then provided in the configuration,
as `datasource` sub-configuration. A datasource contains the pointers to the paths of
the manifest files for each Music Multiple stage (or split) along with additional information
(eg. maximum sample rate to use against this dataset). All the datasources are under the
`dset` group config, with a dedicated configuration file for each dataset.

## Getting started

### Example

See the provided example in the directory that provides a manifest to use the example dataset
provided under the [dataset folder](../dataset/example).

The manifest files are stored in the [egs folder](../egs/example).

```shell
egs/
  example/data.json.gz