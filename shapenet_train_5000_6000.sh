#!/bin/bash
#SBATCH --job-name=prune_gs
#SBATCH --output=sbatch_log/prune_gs_5000_6000_%j.out


module load gcc/9.3.0
module load eth_proxy
module load cuda/11.8.0
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda activate feature_3dgs

python -u encode_images.py --backbone clip_vitl16_384 --weights /cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/checkpoints/demo_e200.ckpt --widehead --no-scaleinv --outdir /cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/data/72a74e13c2424c19f2b0736dd4d8afe0/rgb_feature_langseg --test-rgb-dir /cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/data/72a74e13c2424c19f2b0736dd4d8afe0/image --workers 0


python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input /cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/data/72a74e13c2424c19f2b0736dd4d8afe0/image  --output /cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/data/72a74e13c2424c19f2b0736dd4d8afe0/sam_embeddings



python train.py -s /cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/data/db/drjohnson/ -m output/drjohnson -f lseg -r 0 --speedup --iterations 7000