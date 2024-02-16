FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

RUN conda install pyg pytorch-sparse pytorch-cluster pytorch-scatter pytorch-spline-conv -c pyg
RUN pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
