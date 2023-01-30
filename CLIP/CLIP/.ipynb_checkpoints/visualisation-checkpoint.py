import CLIP.clip as clip
import scipy.io
import torch
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from CLIP.Interpretor import Interpretor
import argparse
import torchvision.transforms as transforms
import random
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets import CIFAR100, ImageNet, CIFAR10
import pathlib
from sklearn import metrics 


import warnings

warnings.filterwarnings("ignore")

class StanfordCars(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        self.transform = transform

        self._base_folder = pathlib.Path(root)
        devkit = self._base_folder / "devkit"

        self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
        self._images_base_path = self._base_folder / "cars_test"

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        
        return pil_image, target

def main():
    parser = argparse.ArgumentParser(description='CLIP')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='rollout',
                        choices=['rollout', 'attn_last_layer', 'attn_gradcam', 'generic_attribution', 'ours', 'ours_c'],
                        help='')
    parser.add_argument('--image_path', type=str,
                        default=None,
                        help='')
    parser.add_argument('--perturbation_level', type=list,
                        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        help='')
    parser.add_argument('--neg', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--target', type=bool,
                        default=False,
                        help='')
    
    parser.add_argument('--start_layer', type=int,
                        default=10,
                        help='')
    
    parser.add_argument('--slices', type=int,
                        default=1,
                        help='')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("/root/datasets/models/pretrained_model/ViT-B-32.pt", device=device, jit=False)
    
#     dataset = CIFAR100('/root/datasets/CIFAR100/', train=False, download=False, transform=preprocess)
#     num_samples = 5000
#     random.seed(0)
#     index = random.sample(range(len(dataset)), num_samples)
#     sub_dataset = torch.utils.data.Subset(dataset, indices=index)
#     sample_loader = torch.utils.data.DataLoader(
#         sub_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4
#     )

#     dataset =StanfordCars(args.image_path, transform=preprocess)
#     num_samples = 5000
#     random.seed(0)
#     index = random.sample(range(len(dataset)), num_samples)
#     sub_dataset = torch.utils.data.Subset(dataset, indices=index)

#     sample_loader = torch.utils.data.DataLoader(
#         sub_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4
#     )

    dataset = ImageNet('/root/datasets/ImageNet', split='val', download=False, transform=preprocess)
    num_samples = 5000
    import random
#     random.seed(0)
#     index = random.sample(range(len(dataset)), num_samples)
    index = np.load('../pert_index.npy')
    sub_dataset = torch.utils.data.Subset(dataset, indices=index)
    sample_loader = torch.utils.data.DataLoader(
        sub_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    interpret = Interpretor(model, device)
    texts = clip.tokenize([f"a photo of a {c}" for c in dataset.classes]).to(device)
    curr_pert_sum = [0]*len(args.perturbation_level)
    logits = torch.zeros(1, 1000).to(device)
    
    iterator = tqdm(sample_loader)
    
    for batch_idx, (data, target) in enumerate(iterator):
        
        data = data.to(device)
        target = target.to(device)
        
        if args.target:
            text = texts[target]
        else:
            for i in range(args.slices):
                with torch.no_grad():
                    logits_per_image, logits_per_text = model(data, texts[int((1000/args.slices)*i): int((1000/args.slices)*(i+1))])
                logits[:, int((1000/args.slices)*i): int((1000/args.slices)*(i+1))] = logits_per_image
            index = torch.argmax(logits.softmax(dim=-1).detach())
            text = texts[index].unsqueeze(0)
            
        if args.method == 'attn_gradcam':
            R = interpret.generate_cam_attn(data, text)
        elif args.method == 'attn_last_layer':
            R = interpret.generate_raw_attention(data, text)
        elif args.method == 'rollout':
            R = interpret.generate_rollout(data, text)
        elif args.method == "generic_attribution":
            R = interpret.generate_generic_attribution(data, text)
        elif args.method == "ours":
            R = interpret.generate_ours(data, text, num_layers=args.start_layer)
        elif args.method == "ours_c":
            R = interpret.generate_ours_c(data, text, num_layers=args.start_layer)
        
        R = torch.nn.functional.interpolate(R.reshape(1, 1, 7, 7), scale_factor=32, mode='bilinear').cuda()
        R = (R - R.min()) / (R.max() - R.min())
        R = R.reshape(1, -1)
    
        if args.neg:
            R = -R

        org_shape = data.shape
        num_correct_pertub = np.zeros(len(args.perturbation_level))
        base_size = 224 * 224

        for i in range(len(args.perturbation_level)):

            _images = data.clone()

            _, idx = torch.topk(R, int(base_size * args.perturbation_level[i]), dim=-1)
            idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
            _images = _images.reshape(1, org_shape[1], -1)
            _images = _images.scatter_(-1, idx, 0)
            _images = _images.reshape(*org_shape)

            for j in range(args.slices):
                with torch.no_grad():
                    logits_per_image, logits_per_text = model(_images, texts[int((1000/args.slices)*j): int((1000/args.slices)*(j+1))])
                logits[:, int((1000/args.slices)*j): int((1000/args.slices)*(j+1))] = logits_per_image
            index = torch.argmax(logits.softmax(dim=-1).detach())
            num_correct_pertub[i]= (target == index).type(target.type()).data.cpu().numpy()
        
        curr_pert_sum += num_correct_pertub

        curr_pert_result = [round(res / (batch_idx+1) * 100, 2) for res in curr_pert_sum]
        iterator.set_description("Acc: {}".format(curr_pert_result))
    
    x = np.arange(0,1.1,0.1)
    y = np.array(curr_pert_result)
    res = metrics.auc(x,y)
    curr_pert_result.append(res)
    np.save(f'res_{args.method}_{args.neg}_{args.target}_{args.start_layer}', np.array(curr_pert_result))
    
    

if __name__ == "__main__":
    main()