FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install PyG with appropriate PyTorch and CUDA versions to match the base image
RUN pip install pyg_lib==0.3.0 \
    torch_scatter==2.1.2 \
    torch_sparse==0.6.18 \
    torch_cluster==1.6.3 \
    torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    && pip install torch-geometric==2.4.0
