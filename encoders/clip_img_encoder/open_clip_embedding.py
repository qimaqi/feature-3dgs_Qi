import torch
from PIL import Image
import open_clip
import os 
import glob

image_dir = '/cluster/work/cvl/qimaqi/3dv_gaussian/feature-3dgs_Qi/data/72a74e13c2424c19f2b0736dd4d8afe0/image/'
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    for image_i in image_paths:
        image = preprocess(Image.open(image_i)).unsqueeze(0)
        # print("image", image.shape) # ([1, 3, 224, 224])
        image_features = model.encode_image(image)
        print("image_features", image_features.shape) # torch.Size([1, 512]) why

 



