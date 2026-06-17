# R2Gen을 활용한 Vision Encoder 및 Text Decoder 아키텍처 비교 실험



<img width="628" height="323" alt="image" src="https://github.com/user-attachments/assets/65836de3-8053-4efc-a525-ddeec8db1e13" />



## Installation

To get started, clone the repository and install dependencies:

```bash
!git clone https://github.com/AI-SJS/AI-SJS-R2gen-experiment.git
%cd AI-SJS-R2gen-experiment
!ls
```

## Datasets
We use datasets (IU X-Ray) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

## Experiment

<details>
<summary><strong>🔷 Encoder Experiment</strong></summary>

## Requirements 

```bash
- nltk
- scikit-image
- matplotlib
- pycocotools
- pandas
- pillow
- tqdm
- numpy
- torch torchvision
- timm
```
## METEOR Download
This project requires the METEOR scorer.

🔗 Download meteor from:
[METEOR Download](https://github.com/zhjohnchan/R2Gen/tree/main/pycocoevalcap/meteor)

After downloading, please place the `meteor` file into the following directory: R2Gen/pycocoevalcap/


## Download R2Gen Encoder (Encoder_experiment)
You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/1E44ufzy6K0IF3UQ0j6vAtydLCdtjnvqt?usp=drive_link).

---

## Run on IU X-Ray (Encoder_experiment)

To train the model on the IU X-Ray dataset, run the following command:

```bash
!python main_train.py \
  --dataset_name iu_xray \
  --ann_path "/content/iu_xray/annotation.json" \
  --image_dir "/content/iu_xray/images" \
  --save_dir "/content/iu_xray/saves/xcit_medium_24_p16_224" \
  --visual_extractor xcit_medium_24_p16_224 \
  --epochs 15
  --seed 9223
  # --visual_extractor resnet101
  # --visual_extractor densenet121
  # --visual_extractor vit_b_16
  # --visual_extractor vit_b_32
  # --visual_extractor xcit_small_12_p16_224
  # --visual_extractor xcit_medium_24_p16_224
```

## Test on IU X-Ray (Encoder_experiment)

To evaluate the trained model on the IU X-Ray dataset, run:

```bash
!python main_test.py \
  --dataset_name iu_xray \
  --ann_path "/content/iu_xray/annotation.json" \
  --image_dir "/content/iu_xray/images" \
  --save_dir "/content/iu_xray/saves/xcit_medium_24_p16_224" \
  --load "/content/iu_xray/saves/xcit_medium_24_p16_224/model_best.pth" \
  --visual_extractor xcit_medium_24_p16_224
```
</details>

<details>
<summary><strong>🔶 Decoder Experiment</strong></summary>

## Requirements 

```bash
- nltk
- scikit-image
- matplotlib
- pycocotools
- pandas
- pillow
- tqdm
- numpy
- torch torchvision
- timm
- "transformers>=4.40" accelerate bitsandbytes sentencepiece
```

## Download R2Gen_Llama-2-7b (Decoder_experiment)
You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/17AkUQcBcZCCGytc4I1SCGgh6Uy_0QFOI?usp=sharing).

## Download R2Gen_MedLlama-2-7b (Decoder_experiment)
You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/1lUlprw217juWu8Lb2o0bZZU3xMNEbybJ?usp=sharing).

## Download R2Gen_MedLlama-2-7b_Q-former (Decoder_experiment)
You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/1Hh4TZujpzftNWc8lI8b9NNRXXaliaeA4?usp=sharing).

## Run on IU X-Ray, R2Gen_Llama-2-7b

To train the model on the IU X-Ray dataset, run the following command:

```bash
!python main_train.py \
  --dataset_name iu_xray \
  --image_dir "/content/drive/MyDrive/iu_xray/images" \
  --ann_path "/content/drive/MyDrive/iu_xray/annotation.json" \
  --save_dir "/content/drive/MyDrive/R2Gen_Llama27b/Results" \
  --visual_extractor resnet101 \
  --epochs 15 \
  --decoder_type llama \
  --llama_model_name "meta-llama/Llama-2-7b-hf" \
  --llama_load_in_8bit \
  --freeze_llama
  --seed 9223
```

## Test on IU X-Ray, R2Gen_Llama-2-7b

To evaluate the trained model on the IU X-Ray dataset, run:

```bash
!python main_test.py \
  --dataset_name iu_xray \
  --image_dir "/content/drive/MyDrive/iu_xray/images" \
  --ann_path "/content/drive/MyDrive/iu_xray/annotation.json" \
  --save_dir "/content/drive/MyDrive/R2Gen_Llama27b/Results" \
  --load "/content/drive/MyDrive/R2Gen_Llama27b/Results/model_best.pth" \
  --visual_extractor resnet101 \
  --decoder_type llama \
  --llama_model_name "meta-llama/Llama-2-7b-hf" \
  --llama_load_in_8bit \
  --freeze_llama
```

## Run on IU X-Ray, R2Gen_MedLlama2-7b

To train the model on the IU X-Ray dataset, run the following command:

```bash
!python main_train.py \
  --dataset_name iu_xray \
  --image_dir "/content/drive/MyDrive/iu_xray/images" \
  --ann_path "/content/drive/MyDrive/iu_xray/annotation.json" \
  --save_dir "/content/drive/MyDrive/R2Gen_MedLlama27b/Results" \
  --visual_extractor resnet101 \
  --epochs 15 \
  --decoder_type llama \
  --llama_model_name "llSourcell/medllama2_7b" \
  --llama_load_in_8bit \
  --freeze_llama
  --seed 9223
```

## Test on IU X-Ray, R2Gen_MedLlama2-7b

To evaluate the trained model on the IU X-Ray dataset, run:

```bash
!python main_test.py \
  --dataset_name iu_xray \
  --image_dir "/content/drive/MyDrive/iu_xray/images" \
  --ann_path "/content/drive/MyDrive/iu_xray/annotation.json" \
  --save_dir "/content/drive/MyDrive/R2Gen_MedLlama27b/Results" \
  --load "/content/drive/MyDrive/R2Gen_MedLlama27b/Results/model_best.pth" \
  --visual_extractor resnet101 \
  --decoder_type llama \
  --llama_model_name "llSourcell/medllama2_7b" \
  --llama_load_in_8bit \
  --freeze_llama
```

## Run on IU X-Ray, R2Gen_MedLlama2-7b_Q-former

To train the model on the IU X-Ray dataset, run the following command:

```bash
!python main_train.py \
  --dataset_name iu_xray \
  --image_dir "/content/drive/MyDrive/iu_xray/images" \
  --ann_path "/content/drive/MyDrive/iu_xray/annotation.json" \
  --save_dir "/content/drive/MyDrive/R2Gen_MedLlama27b_Q-former/Results" \
  --visual_extractor resnet101 \
  --epochs 15 \
  --decoder_type llama \
  --llama_model_name "llSourcell/medllama2_7b" \
  --llama_load_in_8bit \
  --freeze_llama
  --seed 9223
```

## Test on IU X-Ray, R2Gen_MedLlama2-7b_Q-former

To evaluate the trained model on the IU X-Ray dataset, run:

```bash
!python main_test.py \
  --dataset_name iu_xray \
  --image_dir "/content/drive/MyDrive/iu_xray/images" \
  --ann_path "/content/drive/MyDrive/iu_xray/annotation.json" \
  --save_dir "/content/drive/MyDrive/R2Gen_MedLlama27b_Q-former/Results" \
  --load "/content/drive/MyDrive/R2Gen_MedLlama27b_Q-former/Results/model_best.pth" \
  --visual_extractor resnet101 \
  --decoder_type llama \
  --llama_model_name "llSourcell/medllama2_7b" \
  --llama_load_in_8bit \
  --freeze_llama
```

</details>



