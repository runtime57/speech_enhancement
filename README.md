# Multimodal System for Audio-Visual Deepfake Detection

We investigate parameter-efficient modifications of DeepFilterNet by replacing GRU blocks with FastGRNN and exploring alternative convolutional designs. Our approach significantly reduces the model size while preserving comparable speech enhancement quality. See the paper for details.


## Installation

0. Create and install [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment

    ```
    conda create -n speech_enhancement python=3.8 -y
    conda activate speech_enhancement
    ```

1. Clone this repository

    ```
    git clone git@github.com:runtime57/speech_enhancement.git
    cd speech_enhancement
    ```

2. Install all required packages
    ```
    pip install -r requirements.txt
    ```
3. Install pre-commit
    ```
    pre-commit install
    ```

## Experimental setup

For both training and evaluation, we used the [VCTK+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) dataset, which is widely used in speech enhancement research as a standard benchmark, which allows for direct comparison between different methods.

Configurations for all types of used baselines are listed in `src/configs/model`. Feel free to conduct your own experiments by changing used model in `src/configs`.


## Train and inference
 

To train a new model, use the following command:
```
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```
Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments

To run inference (evaluate the model or save predictions):
```
python3 inference.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS  # for metrics
python3 tech_eval.py  # for params, MACs and other technical metrics. Choose the model in tech_eval.yaml
```

use `from_pretrained` field in a `{model_name}.yaml` file to load the checkpoint you need from `checkpoints/{model_name}`.

## Results on VCTK+DEMAND dataset


| Model         | Params (M) | MACs (G) | FLOPs (G) | PESQ | STOI | SI-SNR |
|--------------|-----------:|---------:|----------:|-----:|-----:|-------:|
| Original DFNet | 1.78      | 0.35     | 0.30      | 2.81 | 0.93 | 17.30  |
| Our DFNet      | 1.69      | 0.51     | 0.33      | **2.31** | 0.92 | **17.02** |
| FastDFNet      | 1.04      | 0.34     | **0.39**  | 2.24 | 0.92 | 16.78  |
| Rebalanced-1   | 1.03      | 0.25     | 0.30      | 2.21 | 0.92 | 16.53  |
| Rebalanced-2   | 1.09      | 0.21     | 0.27      | 2.17 | 0.92 | 16.77  |
| Rebalanced-3   | 1.15      | **0.19** | 0.26      | 2.15 | 0.92 | 16.40  |


Best values per column are highlighted in **bold**.
Due to limited computational and storage resources, we did not evaluate our models on the DNS Challenge dataset.

## Future work

- Benchmark on DNS Challenge dataset for better comparability with recent methods
- Explore GAN-based training to improve perceptual quality (e.g., PESQ)