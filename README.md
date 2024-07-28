# Code Appendix of "Zero-Shot Character Identification and Speaker Prediction in Comics via Iterative Multimodal Fusion"
This code appendix includes all codes to reproduce the main experiments of the paper.

This README consists of two parts:
- Codes: briefly explain each file (e.g., which file corresponds to which module).
- How to run: describe how to reproduce the experiments.

# Codes
- `speaker_prediction/`
  - `speaker_prediction.py`
    - Main module of our approach.
    - You can get the correspondence between pseudo codes in the paper (Algorithm 1) and actual codes by reading the function `ZeroShotSpeakerCharacterPredictor.predict_speaker_and_classify_characters()`.
  - `llm_speaker_annotation.py`
    - Speaker prediction module. 
  - `prompts/`
    - LLM prompts used in our experiments. (English translated version is in supplementary pdf.)
  - `character_classification.py`
    - Character identification module.
  - `image_classification/`
    - The codes related to training and testing image classifiers.
- `eval.py`
  - Codes related to running experiments including pre-processing of Manga109 dataset, executing the modules, and saving the experimental results.
- `aggregate_eval.py`
  - Analyze experimental results obtained from eval.py to compute final metrics.     

# How to run

## Prerequisites

- **Manga109 dataset**
    - Download from http://www.manga109.org/en/download.html
    - put at `./data/Manga109_2017_09_28`
- **Manga109 Dialog annotation**
    
    ```
    cd data/
    git clone https://github.com/manga109/public-annotations
    ```
- **Pre-trained models for character classification** 
    ```
    mkdir data/models/
    wget https://ldcwnxk0f27z0wgkkof02ejpxpyglx73.s3.ap-northeast-1.amazonaws.com/models/data70_resnet_square_model.pt -P data/models/
    ```
    

## Environment setup

- Setup python environment
    
    ```
    # Create and activate a Conda environment
    conda create -n speaker-prediction python=3.8
    conda activate speaker-prediction
    # Install required packages
    pip install -r requirements.txt
    ```
    
- Set Open AI API key
    ```
    # Set your OpenAI API key (replace 'sk-xxxxxx' with your actual key)
    export OPENAI_API_KEY=sk-xxxxxx
    ```
    - You can get API key from https://openai.com/blog/openai-api
    - Ensure you use an account where GPT-4 is available.
    - Note that API cost will be charged to your account by running codes.

## Test

Execute `eval.py` for speaker prediction and character identification using the Manga109 dataset.

- Single book
    
    ```
    python eval.py -d data/Manga109_2017_09_28 -n 2 -b YamatoNoHane --exp_name test_single_book # Specify book title as argument '-b'
    ```
    
- Text only (all books of test set)
    
    ```
    python eval.py -d data/Manga109_2017_09_28 -n 1 --exp_name test_text_only # Add argument '-n 1'
    ```
    
- Up to 3rd iteration (all books of test set)
    
    ```
    python eval.py -d data/Manga109_2017_09_28 -n 4 --exp_name test_multimodal # Add argument '-n 4'
    ```
    

## Analysis of results

Analyze your experiment's results using `aggregate_eval.py`.

```
python aggregate_eval.py --exp_names test_multimodal --steps 0 1 2 3 # Specify experiment names and iteration steps that you want to analyze

# single book
python aggregate_eval.py --exp_names test_single_book --steps 0 1 -b YamatoNoHane

```

Metrics for evaluation include:
- **Speaker Prediction**
  - `T_micro`: Micro-average accuracy (main metric)
  - `T_macro`: Macro-average accuracy
  - `T_macro_top5`: Macro-average accuracy for the top 5 characters
  
- **Character Identification**
  - `C_micro`: Micro-average accuracy (main metric)
  - `C_macro`: Macro-average accuracy
  - `C_macro_top5`: Macro-average accuracy for the top 5 characters
