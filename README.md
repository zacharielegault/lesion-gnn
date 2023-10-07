# DRG-NET: A graph neural network for computer-aided grading of diabetic retinopathy

A reimplemetation of the [article](https://rdcu.be/dnENc) by Salam et al.

## Contributing
To contribute to the project, run the following commands:
```bash
git clone
cd DRG-NET
pip install -e ".[all]"
```
A devcontainer setup is also available for VSCode users.

Make sure the [pre-commit hooks](https://pre-commit.com/) are installed:
```bash
pre-commit install
```

To run the tests, run the following command:
```bash
pytest --cov=drgnet --cov-report=html
```
