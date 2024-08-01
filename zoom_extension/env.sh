conda activate nod-pytorch
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_PATH=/opt/rocm
# export HIP_ROOT_DIR=/opt/rocm/hip
export TORCH_TEST_DEVICES=/home/arhakhan/pytorch/zoom_extension/test/pytorch_test_base.py