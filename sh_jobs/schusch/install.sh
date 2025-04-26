#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/convformer0_2layer_acdc_inception_conv_elu_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 120GB


conda create -n feature_3dgs python=3.9 -y

conda activate feature_3dgs


export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html
pip install numpy==1.24.1 tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile open-clip-torch numba open3d git+https://github.com/zhanghang1989/PyTorch-Encoding/ 

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 pytorch-lightning setuptools plyfile pytorch-lightning ftfy regex git+https://github.com/openai/CLIP.git altair tensorboardX tensorboard test-tube wandb torchmetrics scikit-image scikit-learn git+https://github.com/zhanghang1989/PyTorch-Encoding/ timm matplotlib opencv-python nunpy==1.24.1



# cmake .. -DOpenCV_DIR=$CONDA_PREFIX/lib/cmake/opencv4

# conda install -c conda-forge opencv
# conda install -c conda-forge cmake==3.16

# cmake -DOpenCV_DIR=/scratch_net/schusch/qimaqi/miniconda3/envs/renderpy/cmake/opencv4 ..
# /home/qi/miniconda3/envs/your_env/lib/cmake/opencv4