# RL-Align Recipes

[![CPU checks](https://github.com/RL-Align/recipes/actions/workflows/cpu.yml/badge.svg)](https://github.com/RL-Align/recipes/actions/workflows/cpu.yml)

Model-specific guides for RL training, train–rollout consistency and performance
with **RL-Kernel**. Find a model, choose a hardware backend, and follow its recipe.

## Recipes

### DeepSeek

| Model | Backend | Framework | Allocation | Guide |
| --- | --- | --- | --- | --- |
| DeepSeek-V4 · four layers | AMD ROCm | verl + RL-Kernel | Single node · 8 GPUs | [Open recipe](models/deepseek-v4/rocm/README.md) |
| DeepSeek-V4 · four layers | AMD ROCm | verl + RL-Kernel | Four nodes · 32 GPUs | [Open recipe](models/deepseek-v4/rocm/README.md) |

Each recipe states its tested scope and validation status. An available
configuration is not a hardware acceptance result.

## Getting started

```bash
git clone https://github.com/RL-Align/recipes.git
cd recipes
```

1. Open the guide for your model and backend.
2. Follow its image, model and data preparation steps, then configure your nodes.
3. Run the provided commands and check the acceptance result and performance report.

Image downloads, version pins and launch commands are documented in each recipe.
Images, weights, datasets and run artifacts remain outside Git.

## Contributing

Add recipes under `models/<model>/<backend>/`. Keep each guide self-contained,
with its own runtime references, configurations and validation record.
See the [contribution guide](docs/recipe-contract.md) for requirements and CPU checks.

## License

Repository code and documentation are [MIT licensed](LICENSE).
Bundled framework patches retain their [upstream licenses](NOTICE.md).
