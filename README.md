<p align="center">
    <img src="assets/header.png" width="800px"/>
</p>

# Lesion GNN

[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/index.html)
[![Lightning](https://img.shields.io/badge/Lightning_2.2.0+-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)

## Running the code
To run the code, run the following commands:
```bash
git clone
cd lesion-gnn
pip install .
train --config "configs/<config_name>.py"
```

## Contributing
To contribute to the project, run the following commands:
```bash
git clone
cd lesion-gnn
pip install -e ".[all]"
```
A [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) setup is also available for VSCode users.
If you want to use the devcontainer to run experiments, install the [devcontainer cli](https://code.visualstudio.com/docs/devcontainers/devcontainer-cli)
and run the following command:
```bash
devcontainer exec --workspace-folder . train --config configs/<config_name>.py
```

Make sure the [pre-commit hooks](https://pre-commit.com/) are installed:
```bash
pre-commit install
```

Test coverage is currently quite low. Any contributions to increase it are welcome. To run the tests, run the following
command:
```bash
pytest --cov=lesion_gnn
```
