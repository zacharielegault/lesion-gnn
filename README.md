# DRG-NET: A graph neural network for computer-aided grading of diabetic retinopathy

A reimplemetation of the [article](https://rdcu.be/dnENc) by Salam et al. (2022). Given that the original code is not available, this project aims to reimplement the model and reproduce the results.

## Running the code
To run the code, run the following commands:
```bash
git clone
cd DRG-NET
pip install .
train --config "configs/<config_name>.yaml"
```

The [available config](configs) files currently are:
- `aptos.yaml`
- `ddr_aptos_lesions.yaml`

## Contributing
To contribute to the project, run the following commands:
```bash
git clone
cd DRG-NET
pip install -e ".[all]"
```
A [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) setup is also available for VSCode users.

Make sure the [pre-commit hooks](https://pre-commit.com/) are installed:
```bash
pre-commit install
```

Test coverage is currently quite low. Any contributions to increase it are welcome. To run the tests, run the following command:
```bash
pytest --cov=drgnet
```
