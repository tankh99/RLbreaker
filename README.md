<p align="center">
<h1 align="center"><img align="center">
<strong>When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search
</strong>
</h1>
</p>


## Overview

This is the code repository of the NeurIPS 2024 paper: "When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search". In this paper, we propose an efficient RL-driven jailbreaking attack against large language models. See [paper](https://arxiv.org/abs/2406.08705) for more details.

## Installation

1. Create conda env: 

   ```bash
   conda create --name LLM python=3.8
   ```

2. Install dependencies:

   ```bash
   conda activate LLM
   pip install -r requirements.txt
   ```

## Datasets

The datasets for the harmful question and initial human-written templates are available in `datasets/questions/advbench_questions.csv` and `datasets/prompts/jailbreak-prompt.xlsx`. The questions are obtained from [llm-attacks](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) and the templates are collected from [llm-jailbreak-study](https://sites.google.com/view/llm-jailbreak-study).

## Training

To train the RL agent, run:

```bash
python train_policy.py \
--index=0 \
--env-name=llama3_8b \
--target_model=meta-llama/Meta-Llama-3-8B-Instruct \
--model_path=gpt-3.5-turbo \
--openai_key=[OPENAI_KEY] \
--deepinfra_key=[DEEPINFRA_KEY] \
```

In this example, we select `meta-llama/Meta-Llama-3-8B-Instruct` as the target LLM and train the RL agent, the helper LLM is `gpt-3.5-turbo`. The agent will be saved at `trained_models/ppo/{ENV_NAME}`. The generated templates will be saved at `datasets/prompts_generated`.

## Testing

After training the agent, we test it on unseen questions. 

```bash
python test_policy.py \
--index=0 \
--target_model=meta-llama/Meta-Llama-3-8B-Instruct \
--model_path=gpt-3.5-turbo \
--ckpt_path=trained_models/ppo/llama3_8bbest.pt \
--max_query=10000
--K=1000
--openai_key=[OPENAI_KEY] \
--deepinfra_key=[DEEPINFRA_KEY] \
```

## Code Architecture

```
datasets                            # data directory
├── prompts                         # initial jailbreaking templates
├── questions                       # harmful questions
├── question_unalign_res.csv        # harmful questions and unaligned responses
a2c_ppo_acktr
├── algo                            # implementation of different training algorithms
├── models.py                       # different RL agent
├── envs.py                         # environment wrapper
└── ...
llm_utils
├── creat_model.py                  # create target LLM and helper LLM
├── model_inference.py              # perform LLM inference
└── roberta_utils.py                # GPTFuzzer's reward model related functions
├── utils.py                        # utils functions
├── RL_env.py                       # define RL environment
├── train_policy.py                 # training
├── test_policy.py                  # testing
└── analyze_results.py              # analyze the saved questions and responses
 
```

## Citation

If you find our work helpful, please cite:

```bibtex
@article{rlbreaker,
  title={When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search},
  author={Chen, Xuan and Nie, Yuzhou and Guo, Wenbo and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2406.08705},
  year={2024}
}
```

## Acknowledgement

This repo is partly based on [gptfuzzer](https://github.com/sherdencooper/GPTFuzz/tree/master), [tempera](https://github.com/tianjunz/TEMPERA).
