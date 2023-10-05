FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Install PyG with appropriate PyTorch and CUDA versions to match the base image (2.0.0 and 11.7 respectively)
RUN pip install pyg_lib==0.2.0 \
    torch_scatter==2.1.1 \
    torch_sparse==0.6.17 \
    torch_cluster==1.6.1 \
    torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.0.0+cu117.html \
    && pip install torch-geometric
