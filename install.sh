#! /usr/bin/env bash
pip install -e '.[all]'
pip install -e ./fundus-datamodules
pip install -e ./masked-vit

# Deal with opencv-python-headless
pip freeze | grep -E 'opencv(-contrib)?-python(-headless)?' | xargs pip uninstall -y  # Uninstall everything
pip install opencv-python-headless  # Reinstall the headless version

pre-commit install
