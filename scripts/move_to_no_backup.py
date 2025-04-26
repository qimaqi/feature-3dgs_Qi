import os 
import shutil
import tqdm 

data_root = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/'
tgt_root = '/srv/beegfs02/scratch/qimaqi_data/data/scannet_full/data'
for scene in tqdm.tqdm(os.listdir(data_root)):
    scene_path = os.path.join(data_root, scene)
    if not os.path.exists(os.path.join(scene_path, "dslr", "feature_3dgs", "rgb_feature_langseg")):
        print(f"Skipping {scene} as it does not contain the required directory.")
        continue
    # Delete
    print(" Deleting ", os.path.join(scene_path, "dslr", "feature_3dgs"))
    shutil.rmtree(os.path.join(scene_path, "dslr", "feature_3dgs"), ignore_errors=True)

    # Create the destination directory if it doesn't exist
    # os.makedirs(os.path.join(tgt_root, scene), exist_ok=True)
    # Move the directory
    # print(" Moving ", os.path.join(scene_path, "dslr", "feature_3dgs"), " to ", os.path.join(tgt_root, scene, "feature_3dgs"))
    # shutil.move(os.path.join(scene_path, "dslr", "feature_3dgs"), os.path.join(tgt_root, scene, "feature_3dgs"))
    