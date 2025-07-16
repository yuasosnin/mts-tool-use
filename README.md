# Tool Use Assignemnt

## Installation Guide

1. Install requirements:

`pip install -r requirements.txt`

2. Download datasets:

`huggingface_hub download gorilla-llm/Berkeley-Function-Calling-Leaderboard`

`huggingface_hub download Salesforce/xlam-function-calling-60k`

3. Run interactive chat

With a 7B Qwen model:

`python run.py Qwen/Qwen2.5-7B-Instruct`

With the fine-tuned 0.5B model:

`python run.py yuasosnin/Qwen2-0.5B-Tool`

## Training and Evaluation

To run evaluation, follow `eval.sh` script.

To run training, please follow `train.ipynb`
