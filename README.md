# PyTorch Implementation of [Beyond Intuition: Rethinking Token Attributions inside Transformers]

The code for visulaization of each model is provided in the .ipynb file.

## ViT

```
cd ViT
```

### Section A. Segmentation Results

Example:
```
CUDA_VISIBLE_DEVICES=0 python baselines/ViT/imagenet_seg_eval.py --method  --imagenet-seg-path

```
[Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat).

### Section B. Perturbation Results

Example:
```
CUDA_VISIBLE_DEVICES=0 python baselines/ViT/generate_visualizations.py --method  --imagenet-validation-path --vis-class
```

Now to run the perturbation test run the following command:
```
CUDA_VISIBLE_DEVICES=0 python baselines/ViT/pertubation_eval_from_hdf5.py --method --neg --vis-class
```

Notice that you can use the `--neg` argument to run either positive or negative perturbation.

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

### Section A. Language Reasoning
Example:
```
CUDA_VISIBLE_DEVICES=0 python BERT_rationale_benchmark/models/pipeline/bert_pipeline.py --data_dir data/movies/ --output_dir bert_models/movies/ --model_params BERT_params/movies_bert.json --method ours
```

In order to run f1 test with k, run the following command:
```
python BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/<method_name>/identifier_results_k.json
```

### Section B. Perturbation Test
Example:
```
CUDA_VISIBLE_DEVICES=0 python BERT_rationale_benchmark/models/pipeline/bert_pertub_movie.py --data_dir data/movies/ --output_dir bert_models/movies/ --model_params movies_bert.json --method ours
```

For 20ng Dataset, before runing the following command, the code for finetuning is provided in BERT_rationale_benchmark/models/pipeline/bert_finetuning_20ng.py.

```
CUDA_VISIBLE_DEVICES=0 python BERT_rationale_benchmark/models/pipeline/bert_pertub_20ng.py --model_params BERT_params/movies_bert.json --method ours
```

## CLIP

```
cd CLIP
```

### Section A. Perturbation Test
Example:
```
CUDA_VISIBLE_DEVICES=0 python CLIP/visualisation.py --image_path /root/datasets/ImageNet/ --method ours
```

