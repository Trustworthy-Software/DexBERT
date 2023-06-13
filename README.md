# DexBERT
Effective, Task-Agnostic and Fine-grained Representation Learning of Android Bytecode

# Environment

  - Java 11.0.11
  - Python 3.7.11
  - numpy 1.19.5
  - torch 1.7.1
  - torchvision 0.8.2
  - ptflops 0.6.8
  - tensorflow 2.6.0
  - tensorboard 2.7.0
  - scikit-learn 1.0.2

# Usage

## Instruction
  -  For most users, if you just want to use a pre-trained DexBERT to generate class features for your own Android analysis tasks, please skip the pretraining stage and download our pre-trained DexBERT model using the link provided below.
  
  - For readers who want to replicate our experiments, please follow the steps below to pre-train a DexBERT model and apply it in malicious code localization, app defect detection, and component type classification.

  - Please find some smali examples in the folder './Data/examples'.

## DexBERT Pre-training
  - Data preparation: 
    - First, find apk hash list at: ```Data/data/pretraining_apks.txt```
    - Second, download and process APKs: ```python data4pretraining.py -d apk_dir -l apk_hash_list -cp cpu_number```

  - Start pre-training: 
    - ```sh pretrainDexBERT.sh```

  - Infer a pre-trained model: 
    - ```python InferBERT.py --model_cfg config_file_path --data_file pre-processed_data_file --model_file pre-trained_model_file --vocab vocabulary_path```
  
  - Download our pre-trained DexBERT model with this link: https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view?usp=sharing

## Malicious Code Localization
  - Data preparation: 
    - First, download APKs and ground-truth with link: https://sites.google.com/view/mkldroid/dataset-and-results
    - Second, extract Smali instructions: ```python data4malice.py```

  - Training & Evaluation:
    - ```python MaliciousCodeLocalization.py```

## App Defect Detection
  - Data preparation:
    - First, download the APKs with link: https://github.com/breezedong/DNN-based-software-defect-prediction; labels for defective smali files are provided in ```Data/data/defect_labels```
    - Second, extract Smali instructions and generate sample list: ```python data4defect.py```

  - Training & Evaluation:
    - ```python AppDefectDetection.py```

## Component Type Classification
  - Data preparation:
    - ```cd Data & python data4component.py```
  - Training & Evaluation:
    - ```cd Models & python ComponentTypeClassification_FirstState768.py```

## Compute Model Flops
  - ```python count_flops.py```

## Notes:
  - Embedding Size
    - To find a reasonable trade-off between model computation cost and performance, we conducted an ablation study exploring the impact of DexBERT embedding size on three downstream tasks. The experiments contain three different sizes for the hidden embedding of the AutoEncoder (AE), specifically 256, 128, and 64. Additionally, we evaluated the performance by directly utilizing the first state vector of the raw DexBERT embedding, which has a size of 768, without applying any dimension reduction from the AutoEncoder. 
    - The experimental results reveal that in the task of Malicious Code Localization, a decrease in vector size does not lead to a significant loss in the performance, until the size is reduced to 128. As for the tasks of Defect Detection and Component Type Classification, the experimental results demonstrate that a larger embedding size resulted in a considerable improvement in performance. However, a size of 128 also offered a solid trade-off for these two tasks, supporting satisfactory performance with a metric score exceeding 0.9.
  - AutoEncoder Module: We considered two potential inputs for the AutoEncoder: the full DexBERT embedding (512x768), and the first state vector of the embedding (size 768). From our observations, these inputs yielded similar performance. However, using the first state vector of the embedding was found to be more efficient, leading to faster convergence during fine-tuning for downstream tasks. Therefore, we use the first state vector as the default input for AutoEncoder.