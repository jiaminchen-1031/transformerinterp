# Beyond Intuition: Rethinking Token Attributions inside Transformers

<div align="left">
<img src=roadmap.png width=600/>
</div>

## Abstract

The multi-head attention mechanism, or rather the Transformer-based models, have always been under the spotlight, not only in the domain of text processing, but also for computer vision. Several works have recently been proposed around exploring the token attributions along the intrinsic decision process. However, the ambiguity of the expression formulation can lead to an accumulation of error, which makes the interpretation less trustworthy and less applicable to different variants. In this work, we propose a novel method to approximate token contributions inside Transformers. We start from the partial derivative to each token, divide the interpretation process into ***attention perception*** and ***reasoning feedback*** with the chain rule and explore each part individually with explicit mathematical derivations. In attention perception, we propose the head-wise and token-wise approximations in order to learn how the tokens interact to form the pooled vector. As for reasoning feedback, we adopt a noise-decreasing strategy by applying the integrated gradients to the last attention map. Our method is further validated qualitatively and quantitatively through the faithfulness evaluations across different settings: single modality (BERT and ViT) and bi-modality (CLIP), different model sizes (ViT-L) and different pooling strategies (ViT-MAE) to demonstrate the broad applicability and clear improvements over existing methods. 

**The full paper can be found [here](https://openreview.net/pdf?id=rm0zIzlhcX).**

<div align="left">
<img src=example.png width=400/>
</div>


## Requirements

Run the following command to create the conda environment:

```
conda env create -f torch1.10.1.yml
conda activate torch1.10.1
```

The code for the visulaizations of CLIP and ViT is provided in the corresponding .ipynb file.

## ViT

```
cd ViT
```

### Section A. Reproducing the segmentation test

[Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat).

#### Example for **Ours-H**

Run the following command to reproduce the results of our head-wise method.

```
CUDA_VISIBLE_DEVICES=0 python baselines/ViT/imagenet_seg_eval.py --method ours --imagenet-seg-path path-to-gtsegs_ijcv.mat
```
Results of other methods can be reproduced by modifying `--method` argument (`ours` for head-wise and `our-c` for token-wise).


### Section B. Reproducing the perturbation test

[Link to download ImageNet Val](https://www.image-net.org/download.php).

#### Example for **Ours-H**

First, run the following command to generate the explanation results:
```
CUDA_VISIBLE_DEVICES=0 python baselines/ViT/generate_visualizations.py --method ours --imagenet-validation-path path-to-imagenet-validation
```

Now to run the perturbation test with following command:
```
CUDA_VISIBLE_DEVICES=0 python baselines/ViT/pertubation_eval_from_hdf5.py --method ours
```

Notice that you can use the `--neg` argument to run either positive or negative perturbation and `--vis-class` argument to choose target or top class.

## BERT

```
cd BERT
```

1. Download the pretrained weights for movie reviews:

- Download `classifier.zip` from https://drive.google.com/file/d/1kGMTr69UWWe70i-o2_JfjmWDQjT66xwQ/view?usp=sharing
- mkdir -p `./bert_models/movies`
- unzip classifier.zip -d ./bert_models/movies/

2. Download the dataset pkl file:

- Download `preprocessed.pkl` from https://drive.google.com/file/d/1-gfbTj6D87KIm_u1QMHGLKSL3e93hxBH/view?usp=sharing
- mv preprocessed.pkl ./bert_models/movies

3. Download the dataset:

- Download `movies.zip` from https://drive.google.com/file/d/11faFLGkc0hkw3wrGTYJBr1nIvkRb189F/view?usp=sharing
- unzip movies.zip -d ./data/

### Section A. Reproducing the language reasoning test

#### Example for **Ours-H**

First, run the following command to generate the explanation results:

```
CUDA_VISIBLE_DEVICES=0 python BERT_rationale_benchmark/models/pipeline/bert_pipeline.py --data_dir data/movies/ --output_dir bert_models/movies/ --model_params BERT_params/movies_bert.json --method ours
```

In order to run f1 test with k, run the following command:
```
python BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/ours/identifier_results_k.json
```

### Section B. Reproducing the perturbation test
#### Example for **Ours-H**

Run the following command for the perturbation test on Movie Reviews:
```
CUDA_VISIBLE_DEVICES=0 python BERT_rationale_benchmark/models/pipeline/bert_pertub_movie.py --data_dir data/movies/ --output_dir bert_models/movies/ --model_params movies_bert.json --method ours
```

For 20ng Dataset, before the test, prepare the finetuned checkpoint with [bert_finetuning_20ng.py](https://github.com/jiaminchen-1031/transformerinterp/edit/master/BERT/BERT_rationale_benchmark/models/pipeline/bert_finetuning_20ng.py).

Then run the following command for the perturbation test on 20ng Dataset:

```
CUDA_VISIBLE_DEVICES=0 python BERT_rationale_benchmark/models/pipeline/bert_pertub_20ng.py --model_params BERT_params/movies_bert.json --method ours
```

## CLIP

```
cd CLIP
```

### Section A. Reproducing the perturbation test
#### Example for **Ours-H**

Run the following command for the perturbation test on ImageNet validation set:
```
CUDA_VISIBLE_DEVICES=0 python CLIP/visualisation.py --image_path path-to-imagenet-val --method ours
```

## Citing

If you find this project useful in your research, please consider cite:

```
@article{
chen2022beyond,
title={Beyond Intuition: Rethinking Token Attributions inside Transformers},
author={Jiamin Chen and Xuhong Li and Lei Yu and Dejing Dou and Haoyi Xiong},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=rm0zIzlhcX},
note={}
}
```

## Credits
The pytorch implementation is based on [Transformer Interpretability Beyond Attention Visualization](https://github.com/hila-chefer/Transformer-Explainability). Thanks for their brilliant work !
