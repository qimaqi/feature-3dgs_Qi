from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

import argparse
import os
from PIL import Image
import sklearn
import sklearn.decomposition
import time 


parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)
    pca = None 

    for t in targets:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        # image = cv2.imread(t) # (1423, 1908, 3)
        # print("image", image.shape, image.min(), image.max())
        # rgba 
        img = Image.open(t)
        img_np = np.array(img) / 255.
        if img_np.shape[-1] == 4:
            img_np = img_np[...,:3]*img_np[...,-1:] + (1.-img_np[...,-1:])
        # go back to cv2 image
        image = (img_np*255).astype(np.uint8)

        print("image processed", image.shape, image.min(), image.max())


        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        predictor.set_image(image)
        image_embedding_tensor = torch.tensor(predictor.get_image_embedding().cpu().numpy()[0])
        ###
        print("embedding shape origin: ", image_embedding_tensor.shape)
        img_h, img_w, _ = image.shape
        _, fea_h, fea_w = image_embedding_tensor.shape
        cropped_h = int(fea_w / img_w * img_h + 0.5)
        image_embedding_tensor_cropped = image_embedding_tensor[:, :cropped_h, :]
        print("embedding shape: ", image_embedding_tensor.shape)
        print("image_embedding_tensor_cropped: ", image_embedding_tensor_cropped.shape)
        torch.save(image_embedding_tensor_cropped, os.path.join(args.output, f"{img_name}_fmap_CxHxW.pt"))
        # save feature map of sam visualization

        # fmap = image_embedding_tensor_cropped
        feature_dim = image_embedding_tensor_cropped.shape[0]
     
        if pca is None:
            print("calculate PCA based on 1st image", img_name)
            pca = sklearn.decomposition.PCA(3, random_state=42)
            
            fmap = image_embedding_tensor_cropped.permute(1, 2, 0).reshape(-1, feature_dim).cpu().numpy()

            f_samples = fmap[::3] # downsample
            transformed = pca.fit_transform(f_samples)
            print(pca)
            print("pca.explained_variance_ratio_", pca.explained_variance_ratio_.tolist())
            print("pca.singular_values_", pca.singular_values_.tolist())
            feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
            feature_pca_components = torch.tensor(pca.components_).float().cuda()
            q1, q99 = np.percentile(transformed, [1, 99])
            feature_pca_postprocess_sub = q1
            feature_pca_postprocess_div = (q99 - q1)
            print(q1, q99)
            del f_samples

            torch.save({"pca": pca, "feature_pca_mean": feature_pca_mean, "feature_pca_components": feature_pca_components,
                        "feature_pca_postprocess_sub": feature_pca_postprocess_sub, "feature_pca_postprocess_div": feature_pca_postprocess_div},
                        os.path.join(args.output, "pca_dict.pt"))

        start = time.time()
        print("image_embedding_tensor_cropped", image_embedding_tensor_cropped.shape)
        print("feature_pca_mean", feature_pca_mean.shape)
        print("feature_pca_components", feature_pca_components.shape )
        print("feature_pca_postprocess_sub", feature_pca_postprocess_sub.shape)
        print("feature_pca_postprocess_div", feature_pca_postprocess_div.shape)

        image_embedding_tensor_cropped = image_embedding_tensor_cropped.float().cuda()
        vis_feature = ( image_embedding_tensor_cropped.permute(1, 2, 0).reshape(-1, feature_dim) - feature_pca_mean[None, :]) @ feature_pca_components.T
        vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
        vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fea_h, fea_w, 3)).cpu()
        Image.fromarray((vis_feature.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.output, img_name + "_feature_vis.png"))
        #print(time.time() - start)
        #print("done imgsave")


        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2


# checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=checkpoint)
# sam.to(device='cuda')
# predictor = SamPredictor(sam)

# image = cv2.imread("test/images/IMG_20220408_142309.png")
# predictor.set_image(image)
# image_embedding = predictor.get_image_embedding().cpu().numpy()
# print("embedding shape: ", image_embedding.shape)
# np.save("test/embedding.npy", image_embedding)