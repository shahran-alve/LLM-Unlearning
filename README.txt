Soft Prompting for Unlearning

This repository contains a lightweight reproduction pipeline for the paper
Soft Prompting for Unlearning in Large Language Models.

What is included
- QLoRA base training
- Unlearning baselines:
  - gradient_ascent
  - random_label
  - gradient_ascent_descent
  - gradient_ascent_kl
- SPUL prompt-tuning unlearning
- Unified evaluation scripts
- Result summaries for SST-2 and Yelp

Environment setup
1. Install Miniconda if needed.
2. Create the conda environment:
   conda create -n llm-unlearning python=3.10 -y
3. Activate it:
   conda activate llm-unlearning
4. Install dependencies:
   python -m pip install -r requirements.txt

Package manifest
- See project_environment.json for the package list used in the working conda environment.
- Do not commit .venv or conda env folders to git.

Datasets
- SST-2: karuna-bhaila/Unlearning_SST2
- Yelp polarity: karuna-bhaila/Unlearning_Yelp_Polarity

How to run
1. Train the base model with QLoRA.
   Example:
   python qlora.py --dataset sst2 --model_checkpoints sshleifer/tiny-gpt2 --max_length 256 --lr 2e-4 --train_batch_size 2 --eval_batch_size 2 --num_epochs 1 --weight_decay 0.001 --max_train_retain_samples 800 --max_train_forget_samples 200 --max_test_samples 200 --output_path qlora_checkpoints/tiny-gpt2-qlora-sst2

2. Train unlearning baselines.
   Example:
   python baselines.py --dataset sst2 --model_checkpoints qlora_checkpoints/tiny-gpt2-qlora-sst2 --unlearn_method gradient_ascent --max_length 256 --lr 2e-4 --train_batch_size 2 --eval_batch_size 2 --num_epochs 1 --max_train_retain_samples 800 --max_train_forget_samples 200 --max_test_samples 200 --output_path unlearn_checkpoints/ga_tiny_sst2

3. Train SPUL.
   Example:
   python spul.py --dataset sst2 --model_checkpoints qlora_checkpoints/tiny-gpt2-qlora-sst2 --max_length 256 --lr 2e-4 --train_batch_size 2 --eval_batch_size 2 --num_epochs 1 --weight_decay 0.001 --forget_size 1.0 --ptuning_num_tokens 10 --ptuning_hidden_size 64 --alpha 0.5 --beta 0.1 --max_train_retain_samples 800 --max_train_forget_samples 200 --max_test_samples 200 --output_path unlearn_checkpoints/spul_tiny_sst2

4. Evaluate all checkpoints.
   Example:
   python evaluate_models.py --dataset karuna-bhaila/Unlearning_SST2 --qlora_path qlora_checkpoints/tiny-gpt2-qlora-sst2 --spul_path unlearn_checkpoints/spul_tiny_sst2 --ga_path unlearn_checkpoints/ga_tiny_sst2 --rl_path unlearn_checkpoints/rl_tiny_sst2 --gagd_path unlearn_checkpoints/gagd_tiny_sst2 --gakl_path unlearn_checkpoints/gakl_tiny_sst2 --output_json results_summary_sst2.json

Results
- SST-2 metrics: results_summary_sst2.json
- Yelp metrics: results_summary_yelp.json
- Combined metrics: results_summary.json

Notes
- The repository was validated on a Windows CPU-only machine.
- The current results use a lightweight tiny-gpt2 setup so the pipeline is reproducible without large GPU requirements.
- If you want paper-scale runs, replace the model checkpoints with the original Llama-2 or OPT checkpoints and rerun the same scripts.
