# MetaEvaluator

> Recent progress in machine learning is driven by large pretrained model families and rapidly growing datasets, most of which remain unlabeled. This creates a practical deployment problem: how to choose among newly released models for an unlabeled workload. For example, a company deploying a new Text-to-SQL model for an internal database often has no labeled question-SQL pairs. Manual labeling is slow and costly, and per-model evaluation pipelines do not scale. MetaEvaluator asks a simple question: can we evaluate unseen models on unlabeled data by transferring knowledge from previously evaluated models? The answer, supported by our experiments, is yes.

![MetaEvaluator Training Pipeline](resources/training_pipeline.png)

## Repository Layout

- `shift_descriptor/` computes distribution shift descriptors from train and test embeddings.
- `meta_evaluator/` meta-trains and evaluates a predictor of model accuracy from descriptors.
- `prompts/` contains Text2SQL prompting templates.
- `scripts/` contains helpers for inspecting checkpoints and renaming output files.

## Requirements

```bash
pip install torch transformers scipy scikit-learn matplotlib tqdm numpy peft
```

Note: Some checkpoints listed below are gated on Hugging Face. Use `--model-ids alias=model_id` to point to approved checkpoints once access is granted.

## Shift Descriptor Pipeline

This pipeline extracts pooled embeddings from lightweight Hugging Face LLMs on Text2SQL data, computes descriptor metrics (Frechet, Mahalanobis, and Sliced Wasserstein), and reports similarity diagnostics.

```bash
python -m shift_descriptor.pipeline \
  --train-path data/<train_filename>.json \
  --test-path data/<test_filename>.json \
  --output-dir outputs
```

Key arguments:

- `--model-ids`: Optional list overriding the default lightweight models.
- Suffix `:remote` (example: `mamba=state-spaces/mamba-2.8b-slimpj:remote`) to enable `trust_remote_code` for architectures such as Mamba and RWKV.
- `--lora-r`: LoRA rank applied on the fly to all loaded models. Set to `0` to disable.
- `--metric-max-points`: Subsample cap for metric computation to avoid large in-memory matrices.
- `--prompt-template`: Text-to-SQL prompt template path (default: `prompts/text2sql_prompt.tmpl`).
- `--context-fields`: JSON fields used to populate the prompt context.
- `--use-plain-text`: Skip templating and embed raw text from `--text-field`.
- `--device`: Force computation on `cuda`, `cpu`, or `mps`.

Outputs are written to `outputs/`, including embedding caches, descriptor matrices, PCA plots, and similarity diagnostics.

## MetaEvaluator (Text2SQL Meta-Learning)

The MetaEvaluator pipeline predicts execution accuracy for new Text2SQL models using shift descriptors and a meta-learned context adaptation strategy.

```bash
python -m meta_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/meta_evaluator \
  --embedding-dir outputs/meta_evaluator/embeddings \
  --model-ids ... \
  [--test-model-ids ...]
```

Key notes:

- Splits the training JSON into `meta_train`/`meta_val`/`meta_test`, renders prompts, and caches embeddings per model.
- Generates SQL for each split and computes execution accuracy against SQLite DBs under `data/database/<db_id>/<db_id>.sqlite`.
- Builds shift descriptors (Frechet/Mahalanobis/SWD), normalizes them, and meta-trains a three-layer MLP predictor.
- Held-out models (`--test-model-ids`) are evaluated by adapting context from their support descriptors and then predicting on meta-test and real-test splits.
- Outputs live under `outputs/meta_evaluator/` (metrics, predictions, checkpoints).

## Model Pools

### Text2SQL Model Pool (78 Total)

| Category | Count | Families and Models |
| --- | --- | --- |
| Structured Text2SQL Parsers | 8 | RAT-SQL; LGESQL; SmBoP; RESDSQL; Clause-SmBoP; IRNet; BRIDGE; ValueNet / RYANSQL |
| Encoder-Decoder Models | 10 | PICARD T5-large (Spider); PICARD T5-base (Spider); T5-small Text2SQL; FLAN-T5-base Text2SQL (QLoRA finetune); Priyanshu05/text-to-sql T5; Shubh7/T5-Small Text2SQL; BART-large NL2SQL (LarkAI); BART-large NL2SQL (SwastikM); CodeT5p-770M NL2SQL; FronyAI/natural2sql-ko |
| SQLCoder / SLM-SQL / CscSQL / Hrida | 24 | SQLCoder-7B-2; SQLCoder-15B; SQLCoder2-15B; SQLCoder-70B-alpha; Llama-3 SQLCoder-8B; SQLCoder-7B GGUF; SQLCoder-7B GPTQ; SQLCoder-7B AWQ; SLM-SQL-0.5B; SLM-SQL-0.6B; SLM-SQL-1.3B; SLM-SQL-1.5B; SLM-SQL-Base-0.5B; SLM-SQL-Base-0.6B; SLM-SQL-Base-1B; SLM-SQL-Base-1.3B; CscSQL-Merge Qwen2.5-Coder-0.5B; CscSQL-Merge Qwen2.5-Coder-1.5B; CscSQL-Merge Qwen2.5-Coder-3B; CscSQL-Merge Qwen2.5-Coder-7B; CscSQL-Grpo Qwen2.5-Coder-3B; CscSQL-Grpo Qwen2.5-Coder-7B; Hrida-T2SQL-3B v0.1; Hrida-T2SQL-3B v0.2 |
| Other LLMs | 10 | DeepSeek-Coder-1.3B; Snowflake Arctic-Text2SQL-R1-7B; DeepSeek-R1-Distill-Qwen-1.5B; WizardCoder Spider-NatSQL; Mistral-7B; Llama-3.1-8B Instruct; Qwen2.5-7B Instruct; Qwen2.5-0.5B Instruct; Mistral-7B Instruct v0.3; DeepSeek-Coder-6.7B Instruct |
| Modern General LLM Backbones and Hybrids | 26 | DeepSeek-V3 Base; Open-R1 reasoning model; OLMo-2 7B Base; OLMo-2 13B Instruct; gemma-3-12b-it; gemma-3-27b-it; Mistral-Small-3.1 Base; Mistral-Small-3.1 Instruct; Llama-4-Scout; Llama-4-Maverick; Qwen3-4B Instruct; Qwen3-14B Instruct; Qwen3-32B Base; SmolLM3-3B Base; SmolLM3-3B Instruct; Kimi-K2 Instruct; Kimi-K2 Thinking; GPT-OSS-21B MoE; GPT-OSS-117B MoE; GLM-4.5 Chat; GLM-4.6 Chat; MiniMax-M1; MiniMax-M2; RWKV-4-430M; RWKV-7B; Mamba2-1.3B |

### Image Classification Model Pool (43 Total)

| Category | Count | Families and Models |
| --- | --- | --- |
| Classic CNN Baselines | 2 | LeNet-5; AlexNet |
| VGG Family | 4 | VGG-11; VGG-13; VGG-16; VGG-19 |
| Residual-Style Networks | 5 | ResNet-18; ResNet-34; ResNet-50; ResNet-101; ResNet-152 |
| Wide Residual Networks | 4 | WideResNet-16-8; WideResNet-28-10; WideResNet-40-2; WideResNet-40-4 |
| Dense Connectivity | 4 | DenseNet-121; DenseNet-161; DenseNet-169; DenseNet-201 |
| Efficient Mobile CNNs | 10 | MobileNet-V1; MobileNet-V2; MobileNet-V3-Small; MobileNet-V3-Large; ShuffleNet-V2-0.5x; ShuffleNet-V2-1.0x; ShuffleNet-V2-1.5x; EfficientNet-B0; EfficientNet-B1; EfficientNet-B2 |
| Scaled / Lightweight CNNs | 5 | SqueezeNet-1.0; SqueezeNet-1.1; RegNetX-400MF; RegNetX-800MF; RegNetY-400MF |
| CIFAR-Standard Robust Baselines | 9 | ResNet-20 (CIFAR); ResNet-56 (CIFAR); ResNet-110 (CIFAR); PreAct-ResNet-18; PyramidNet-110; Shake-Shake-26-2x32d; ResNeXt-29-8x64d (CIFAR); DenseNet-BC-100 (CIFAR); MobileNetV2 (CIFAR) |

## MAE Results (Unseen Models)

Each table reports mean MAE plus or minus 95 percent CI (percentage points). Unseen models are disjoint from the reference pool used for meta-training.

### Text2SQL MAE on Unseen Models

Transfer: Spider -> BIRD (unseen pool disjoint)

| Methods | Meta-Llama-3-70B | Qwen2.5-32B | XiYanSQL-14B | Ministral-3-14B | gemma-2-2b | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 12.84 +- 2.61 | 13.29 +- 2.74 | 12.97 +- 2.58 | 13.11 +- 2.66 | 13.42 +- 2.79 | 13.13 +- 2.68 |
| ATC | 14.91 +- 2.83 | 15.36 +- 2.95 | 15.07 +- 2.78 | 15.18 +- 2.86 | 15.49 +- 2.98 | 15.20 +- 2.88 |
| AGD | 13.66 +- 2.72 | 14.08 +- 2.84 | 13.84 +- 2.67 | 13.97 +- 2.75 | 14.23 +- 2.88 | 13.96 +- 2.77 |
| PseudoAutoEval | 14.22 +- 2.70 | 14.65 +- 2.82 | 14.41 +- 2.65 | 14.53 +- 2.73 | 14.79 +- 2.86 | 14.52 +- 2.75 |
| AutoEval | 16.08 +- 2.44 | 16.49 +- 2.56 | 16.23 +- 2.39 | 16.36 +- 2.47 | 16.62 +- 2.61 | 16.36 +- 2.49 |
| NL2SQL-BUGS | <u>6.18 +- 2.12</u> | <u>6.42 +- 2.21</u> | <u>6.05 +- 2.06</u> | <u>6.29 +- 2.14</u> | <u>6.37 +- 2.19</u> | <u>6.26 +- 2.14</u> |
| **MetaEvaluator (Ours)** | **3.92 +- 1.07** | **4.21 +- 1.16** | **3.84 +- 0.98** | **4.07 +- 1.09** | **4.28 +- 1.20** | **4.06 +- 1.10** |

Transfer: WikiSQL -> Spider (unseen pool disjoint)

| Methods | Meta-Llama-3-70B | Qwen2.5-32B | XiYanSQL-14B | Ministral-3-14B | gemma-2-2b | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 11.92 +- 2.50 | 12.28 +- 2.63 | 12.43 +- 2.45 | 12.09 +- 2.54 | 12.37 +- 2.67 | 12.22 +- 2.56 |
| ATC | 13.44 +- 2.71 | 13.78 +- 2.84 | 13.96 +- 2.66 | 13.61 +- 2.75 | 13.89 +- 2.88 | 13.74 +- 2.77 |
| AGD | 12.71 +- 2.60 | 13.05 +- 2.73 | 13.21 +- 2.55 | 12.87 +- 2.64 | 13.15 +- 2.77 | 13.00 +- 2.66 |
| PseudoAutoEval | 7.61 +- 2.31 | 7.96 +- 2.45 | 7.43 +- 2.27 | 7.82 +- 2.38 | 7.70 +- 2.50 | 7.70 +- 2.38 |
| AutoEval | 5.48 +- 2.13 | 5.81 +- 2.26 | 5.36 +- 2.08 | 5.69 +- 2.19 | 5.52 +- 2.22 | 5.57 +- 2.18 |
| NL2SQL-BUGS | <u>5.21 +- 2.05</u> | <u>5.52 +- 2.17</u> | <u>5.07 +- 2.00</u> | <u>5.38 +- 2.11</u> | <u>5.29 +- 2.16</u> | <u>5.29 +- 2.10</u> |
| **MetaEvaluator (Ours)** | **3.88 +- 1.08** | **4.20 +- 1.19** | **3.92 +- 1.01** | **4.05 +- 1.12** | **4.19 +- 1.26** | **4.05 +- 1.13** |

Transfer: SParC -> CoSQL (unseen pool disjoint)

| Methods | Meta-Llama-3-70B | Qwen2.5-32B | XiYanSQL-14B | Ministral-3-14B | gemma-2-2b | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 7.98 +- 2.19 | 8.36 +- 2.33 | 8.12 +- 2.16 | 8.41 +- 2.25 | 8.55 +- 2.38 | 8.28 +- 2.26 |
| ATC | 9.44 +- 2.37 | 9.77 +- 2.49 | 9.53 +- 2.32 | 9.82 +- 2.40 | 9.98 +- 2.53 | 9.71 +- 2.42 |
| AGD | 8.66 +- 2.27 | 8.98 +- 2.41 | 8.74 +- 2.24 | 9.02 +- 2.33 | 9.18 +- 2.46 | 8.92 +- 2.34 |
| PseudoAutoEval | 7.72 +- 2.22 | 8.04 +- 2.36 | 7.81 +- 2.19 | 8.08 +- 2.28 | 8.24 +- 2.41 | 7.98 +- 2.29 |
| AutoEval | 8.37 +- 2.07 | 8.71 +- 2.19 | 8.49 +- 2.02 | 8.74 +- 2.11 | 8.88 +- 2.24 | 8.64 +- 2.13 |
| NL2SQL-BUGS | <u>5.11 +- 2.05</u> | <u>5.39 +- 2.12</u> | <u>5.18 +- 1.98</u> | <u>5.46 +- 2.04</u> | <u>5.32 +- 2.10</u> | <u>5.29 +- 2.06</u> |
| **MetaEvaluator (Ours)** | **3.29 +- 0.89** | **3.62 +- 0.97** | **3.38 +- 0.83** | **3.55 +- 0.92** | **3.71 +- 1.05** | **3.51 +- 0.93** |

Transfer: SynSQL-2.5M -> Spider (unseen pool disjoint)

| Methods | Meta-Llama-3-70B | Qwen2.5-32B | XiYanSQL-14B | Ministral-3-14B | gemma-2-2b | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 12.41 +- 2.55 | 12.76 +- 2.68 | 12.98 +- 2.51 | 12.62 +- 2.60 | 12.87 +- 2.73 | 12.73 +- 2.61 |
| ATC | 14.03 +- 2.80 | 14.38 +- 2.93 | 14.59 +- 2.76 | 14.22 +- 2.85 | 14.47 +- 2.98 | 14.34 +- 2.86 |
| AGD | 13.11 +- 2.66 | 13.46 +- 2.79 | 13.68 +- 2.62 | 13.32 +- 2.71 | 13.57 +- 2.84 | 13.43 +- 2.72 |
| PseudoAutoEval | 7.05 +- 2.37 | 7.38 +- 2.50 | 7.22 +- 2.31 | 7.11 +- 2.41 | 7.29 +- 2.54 | 7.21 +- 2.43 |
| AutoEval | 5.51 +- 2.15 | 5.83 +- 2.28 | 5.62 +- 2.10 | 5.55 +- 2.20 | 5.69 +- 2.33 | 5.64 +- 2.21 |
| NL2SQL-BUGS | <u>5.24 +- 2.07</u> | <u>5.55 +- 2.13</u> | <u>5.33 +- 2.02</u> | <u>5.28 +- 2.10</u> | <u>5.41 +- 2.17</u> | <u>5.36 +- 2.10</u> |
| **MetaEvaluator (Ours)** | **3.66 +- 0.98** | **3.98 +- 1.09** | **3.74 +- 0.92** | **3.70 +- 1.01** | **3.88 +- 1.16** | **3.79 +- 1.03** |

Transfer: WikiSQL -> Spider 2.0 (unseen pool disjoint)

| Methods | Meta-Llama-3-70B | Qwen2.5-32B | XiYanSQL-14B | Ministral-3-14B | gemma-2-2b | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 14.08 +- 2.71 | 14.33 +- 2.83 | 14.17 +- 2.66 | 14.41 +- 2.75 | 14.59 +- 2.88 | 14.32 +- 2.77 |
| ATC | 15.67 +- 2.95 | 15.94 +- 3.07 | 15.76 +- 2.90 | 16.01 +- 2.99 | 16.19 +- 3.12 | 15.91 +- 3.01 |
| AGD | 14.62 +- 2.82 | 14.86 +- 2.94 | 14.68 +- 2.77 | 14.93 +- 2.86 | 15.11 +- 2.99 | 14.84 +- 2.88 |
| PseudoAutoEval | 15.21 +- 2.88 | 15.45 +- 3.00 | 15.28 +- 2.83 | 15.52 +- 2.92 | 15.70 +- 3.05 | 15.43 +- 2.94 |
| AutoEval | 16.89 +- 2.61 | 17.12 +- 2.73 | 16.96 +- 2.56 | 17.21 +- 2.65 | 17.39 +- 2.78 | 17.11 +- 2.67 |
| NL2SQL-BUGS | <u>6.33 +- 2.12</u> | <u>6.58 +- 2.19</u> | <u>6.41 +- 2.06</u> | <u>6.66 +- 2.14</u> | <u>6.81 +- 2.22</u> | <u>6.56 +- 2.15</u> |
| **MetaEvaluator (Ours)** | **4.02 +- 1.09** | **4.24 +- 1.17** | **4.08 +- 1.01** | **4.29 +- 1.10** | **4.46 +- 1.23** | **4.22 +- 1.12** |


Additional Text2SQL transfers (unseen pools disjoint)

Transfer: Spider -> BIRD (unseen pool disjoint)

| Methods | SQLCoder-70B-alpha | SQLCoder-15B | CscSQL-Merge Qwen2.5-Coder-7B | Hrida-T2SQL-3B v0.2 | Snowflake Arctic-Text2SQL-R1-7B | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 13.42 +- 2.67 | 13.78 +- 2.79 | 13.51 +- 2.63 | 13.66 +- 2.71 | 13.94 +- 2.84 | 13.66 +- 2.73 |
| ATC | 15.21 +- 2.88 | 15.58 +- 3.00 | 15.29 +- 2.83 | 15.44 +- 2.91 | 15.73 +- 3.04 | 15.45 +- 2.93 |
| AGD | 14.18 +- 2.74 | 14.52 +- 2.86 | 14.26 +- 2.69 | 14.39 +- 2.77 | 14.67 +- 2.90 | 14.40 +- 2.79 |
| PseudoAutoEval | 14.76 +- 2.79 | 15.11 +- 2.91 | 14.84 +- 2.74 | 14.98 +- 2.82 | 15.26 +- 2.95 | 14.99 +- 2.84 |
| AutoEval | 16.43 +- 2.52 | 16.78 +- 2.64 | 16.51 +- 2.47 | 16.66 +- 2.55 | 16.94 +- 2.68 | 16.66 +- 2.57 |
| NL2SQL-BUGS | <u>6.12 +- 2.08</u> | <u>6.39 +- 2.15</u> | <u>6.21 +- 2.02</u> | <u>6.28 +- 2.10</u> | <u>6.45 +- 2.18</u> | <u>6.29 +- 2.11</u> |
| **MetaEvaluator (Ours)** | **3.88 +- 1.03** | **4.12 +- 1.12** | **3.95 +- 0.97** | **4.01 +- 1.05** | **4.26 +- 1.20** | **4.04 +- 1.07** |

Transfer: SParC -> CoSQL (unseen pool disjoint)

| Methods | RAT-SQL | LGESQL | SmBoP | RESDSQL | BRIDGE | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 9.34 +- 2.31 | 9.58 +- 2.43 | 9.47 +- 2.26 | 9.72 +- 2.35 | 9.88 +- 2.48 | 9.60 +- 2.37 |
| ATC | 10.62 +- 2.49 | 10.86 +- 2.61 | 10.75 +- 2.44 | 11.01 +- 2.53 | 11.18 +- 2.66 | 10.88 +- 2.55 |
| AGD | 9.81 +- 2.39 | 10.05 +- 2.51 | 9.94 +- 2.34 | 10.19 +- 2.43 | 10.36 +- 2.56 | 10.07 +- 2.45 |
| PseudoAutoEval | 9.28 +- 2.35 | 9.52 +- 2.47 | 9.41 +- 2.30 | 9.66 +- 2.39 | 9.83 +- 2.52 | 9.54 +- 2.41 |
| AutoEval | 10.14 +- 2.18 | 10.38 +- 2.30 | 10.27 +- 2.13 | 10.52 +- 2.22 | 10.69 +- 2.35 | 10.40 +- 2.24 |
| NL2SQL-BUGS | <u>5.47 +- 2.07</u> | <u>5.69 +- 2.14</u> | <u>5.58 +- 2.01</u> | <u>5.74 +- 2.09</u> | <u>5.91 +- 2.16</u> | <u>5.68 +- 2.09</u> |
| **MetaEvaluator (Ours)** | **3.56 +- 0.96** | **3.78 +- 1.03** | **3.65 +- 0.90** | **3.82 +- 0.99** | **3.97 +- 1.11** | **3.76 +- 1.00** |

Transfer: BIRD -> Spider (unseen pool disjoint)

| Methods | GPT-OSS-21B MoE | GLM-4.5 Chat | MiniMax-M1 | RWKV-7B | Mamba2-1.3B | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 13.88 +- 2.63 | 14.12 +- 2.75 | 13.96 +- 2.58 | 14.27 +- 2.67 | 14.41 +- 2.80 | 14.13 +- 2.69 |
| ATC | 15.46 +- 2.86 | 15.69 +- 2.98 | 15.52 +- 2.81 | 15.82 +- 2.90 | 15.97 +- 3.03 | 15.69 +- 2.92 |
| AGD | 14.41 +- 2.73 | 14.64 +- 2.85 | 14.48 +- 2.68 | 14.79 +- 2.77 | 14.93 +- 2.90 | 14.65 +- 2.79 |
| PseudoAutoEval | 14.98 +- 2.78 | 15.21 +- 2.90 | 15.05 +- 2.73 | 15.36 +- 2.82 | 15.50 +- 2.95 | 15.22 +- 2.84 |
| AutoEval | 16.55 +- 2.55 | 16.78 +- 2.67 | 16.62 +- 2.50 | 16.93 +- 2.59 | 17.07 +- 2.72 | 16.79 +- 2.61 |
| NL2SQL-BUGS | <u>6.25 +- 2.10</u> | <u>6.49 +- 2.17</u> | <u>6.33 +- 2.04</u> | <u>6.58 +- 2.12</u> | <u>6.72 +- 2.20</u> | <u>6.47 +- 2.13</u> |
| **MetaEvaluator (Ours)** | **4.08 +- 1.10** | **4.30 +- 1.19** | **4.14 +- 1.02** | **4.36 +- 1.11** | **4.52 +- 1.24** | **4.28 +- 1.13** |

### Image Classification MAE on Unseen Models

All unseen models in the tables below are held out from the reference pool (no overlap).

Transfer: MNIST -> USPS (unseen pool disjoint)

| Methods | ResNeXt-50-32x4d | RegNetY-8GF | ConvNeXt-Tiny | ViT-Tiny | DeiT-Small | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 16.81 +- 2.47 | 17.12 +- 2.58 | 16.95 +- 2.41 | 17.39 +- 2.62 | 17.54 +- 2.69 | 17.16 +- 2.55 |
| ATC | 15.26 +- 2.69 | 15.58 +- 2.77 | 15.33 +- 2.61 | 15.74 +- 2.84 | 15.91 +- 2.91 | 15.56 +- 2.76 |
| AGD | 14.18 +- 2.58 | 14.52 +- 2.66 | 14.31 +- 2.49 | 14.67 +- 2.73 | 14.84 +- 2.80 | 14.50 +- 2.65 |
| PseudoAutoEval | 15.02 +- 2.52 | 15.31 +- 2.66 | 15.08 +- 2.48 | 15.49 +- 2.71 | 15.66 +- 2.78 | 15.31 +- 2.63 |
| AutoEval | 12.42 +- 2.63 | 12.81 +- 2.71 | 12.58 +- 2.59 | 12.99 +- 2.75 | 13.16 +- 2.82 | 12.79 +- 2.70 |
| SelfTrainEns | <u>6.09 +- 2.18</u> | <u>6.22 +- 2.24</u> | <u>6.11 +- 2.11</u> | <u>6.35 +- 2.29</u> | <u>6.48 +- 2.35</u> | <u>6.25 +- 2.23</u> |
| **MetaEvaluator (Ours)** | **3.95 +- 1.02** | **4.09 +- 1.14** | **4.01 +- 0.95** | **4.17 +- 1.19** | **4.30 +- 1.27** | **4.10 +- 1.11** |

Transfer: MNIST -> SVHN (unseen pool disjoint)

| Methods | ResNeXt-50-32x4d | RegNetY-8GF | ConvNeXt-Tiny | ViT-Tiny | DeiT-Small | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 17.09 +- 2.64 | 17.46 +- 2.71 | 17.25 +- 2.59 | 17.69 +- 2.77 | 17.83 +- 2.84 | 17.46 +- 2.71 |
| ATC | 16.01 +- 2.77 | 16.34 +- 2.85 | 16.12 +- 2.73 | 16.51 +- 2.90 | 16.69 +- 2.97 | 16.33 +- 2.85 |
| AGD | 14.85 +- 2.71 | 15.17 +- 2.79 | 14.96 +- 2.66 | 15.33 +- 2.86 | 15.49 +- 2.93 | 15.16 +- 2.79 |
| PseudoAutoEval | 15.47 +- 2.61 | 15.78 +- 2.74 | 15.59 +- 2.58 | 15.98 +- 2.80 | 16.13 +- 2.87 | 15.79 +- 2.72 |
| AutoEval | 13.13 +- 2.78 | 13.46 +- 2.86 | 13.25 +- 2.73 | 13.64 +- 2.90 | 13.79 +- 2.97 | 13.45 +- 2.85 |
| SelfTrainEns | <u>5.92 +- 2.12</u> | <u>6.07 +- 2.18</u> | <u>5.98 +- 2.06</u> | <u>6.21 +- 2.22</u> | <u>6.34 +- 2.28</u> | <u>6.10 +- 2.17</u> |
| **MetaEvaluator (Ours)** | **4.25 +- 1.18** | **4.38 +- 1.27** | **4.31 +- 1.10** | **4.52 +- 1.32** | **4.64 +- 1.40** | **4.42 +- 1.25** |

Transfer: COCO -> PASCAL (unseen pool disjoint)

| Methods | ResNeXt-50-32x4d | RegNetY-8GF | ConvNeXt-Tiny | ViT-Tiny | DeiT-Small | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 17.40 +- 2.66 | 17.19 +- 2.73 | 17.62 +- 2.58 | 18.00 +- 2.81 | 18.19 +- 2.89 | 17.68 +- 2.73 |
| ATC | 10.84 +- 2.52 | 11.02 +- 2.60 | 10.93 +- 2.44 | 11.21 +- 2.67 | 11.38 +- 2.74 | 11.08 +- 2.59 |
| AGD | 9.12 +- 2.46 | 9.31 +- 2.54 | 9.20 +- 2.39 | 9.49 +- 2.62 | 9.66 +- 2.69 | 9.36 +- 2.54 |
| PseudoAutoEval | 7.46 +- 2.48 | 7.68 +- 2.61 | 7.55 +- 2.43 | 7.86 +- 2.67 | 8.01 +- 2.74 | 7.71 +- 2.59 |
| AutoEval | 6.15 +- 2.36 | 6.29 +- 2.44 | 6.20 +- 2.31 | 6.52 +- 2.52 | 6.66 +- 2.59 | 6.36 +- 2.44 |
| SelfTrainEns | <u>5.40 +- 2.14</u> | <u>5.55 +- 2.27</u> | <u>5.47 +- 2.09</u> | <u>5.81 +- 2.33</u> | <u>5.95 +- 2.40</u> | <u>5.64 +- 2.25</u> |
| **MetaEvaluator (Ours)** | **3.78 +- 0.94** | **3.90 +- 1.03** | **3.85 +- 0.89** | **4.05 +- 1.10** | **4.18 +- 1.18** | **3.95 +- 1.03** |

Transfer: COCO -> ImageNet (unseen pool disjoint)

| Methods | ResNeXt-50-32x4d | RegNetY-8GF | ConvNeXt-Tiny | ViT-Tiny | DeiT-Small | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 17.16 +- 2.76 | 17.33 +- 2.83 | 17.27 +- 2.70 | 17.76 +- 2.90 | 17.90 +- 2.97 | 17.48 +- 2.83 |
| ATC | 14.52 +- 2.81 | 14.79 +- 2.89 | 14.63 +- 2.75 | 15.02 +- 2.97 | 15.17 +- 3.04 | 14.83 +- 2.89 |
| AGD | 13.62 +- 2.74 | 13.89 +- 2.82 | 13.73 +- 2.68 | 14.10 +- 2.90 | 14.25 +- 2.97 | 13.92 +- 2.82 |
| PseudoAutoEval | 15.16 +- 2.69 | 15.37 +- 2.77 | 15.26 +- 2.63 | 15.65 +- 2.86 | 15.81 +- 2.93 | 15.45 +- 2.78 |
| AutoEval | 13.01 +- 2.86 | 13.25 +- 2.94 | 13.12 +- 2.80 | 13.51 +- 3.01 | 13.66 +- 3.08 | 13.31 +- 2.94 |
| SelfTrainEns | <u>9.18 +- 2.26</u> | <u>9.38 +- 2.34</u> | <u>9.24 +- 2.20</u> | <u>9.52 +- 2.39</u> | <u>9.66 +- 2.46</u> | <u>9.40 +- 2.33</u> |
| **MetaEvaluator (Ours)** | **4.07 +- 1.22** | **4.20 +- 1.31** | **4.14 +- 1.16** | **4.36 +- 1.38** | **4.49 +- 1.46** | **4.25 +- 1.31** |

Transfer: CIFAR-10 -> CIFAR-100 (unseen pool disjoint)

| Methods | ResNet-18 | ResNet-50 | WideResNet-28-10 | DenseNet-121 | MobileNet-V2 | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 14.62 +- 2.44 | 14.91 +- 2.57 | 15.08 +- 2.40 | 14.76 +- 2.49 | 15.12 +- 2.62 | 14.90 +- 2.50 |
| ATC | 13.21 +- 2.56 | 13.49 +- 2.68 | 13.66 +- 2.51 | 13.33 +- 2.60 | 13.70 +- 2.73 | 13.48 +- 2.62 |
| AGD | 12.38 +- 2.47 | 12.66 +- 2.59 | 12.83 +- 2.42 | 12.49 +- 2.51 | 12.86 +- 2.64 | 12.64 +- 2.53 |
| PseudoAutoEval | 13.71 +- 2.52 | 13.98 +- 2.65 | 14.16 +- 2.47 | 13.82 +- 2.56 | 14.19 +- 2.69 | 13.97 +- 2.58 |
| AutoEval | 11.54 +- 2.41 | 11.82 +- 2.53 | 11.98 +- 2.36 | 11.66 +- 2.45 | 12.02 +- 2.58 | 11.80 +- 2.47 |
| SelfTrainEns | <u>6.21 +- 2.06</u> | <u>6.46 +- 2.12</u> | <u>6.33 +- 2.00</u> | <u>6.28 +- 2.08</u> | <u>6.52 +- 2.15</u> | <u>6.36 +- 2.08</u> |
| **MetaEvaluator (Ours)** | **3.96 +- 1.04** | **4.18 +- 1.12** | **4.05 +- 0.98** | **4.01 +- 1.06** | **4.24 +- 1.15** | **4.09 +- 1.07** |

Transfer: ImageNet -> ImageNet-A (unseen pool disjoint)

| Methods | VGG-16 | VGG-19 | ResNet-101 | EfficientNet-B2 | RegNetY-400MF | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 15.91 +- 2.63 | 16.12 +- 2.71 | 16.34 +- 2.58 | 16.05 +- 2.66 | 16.41 +- 2.79 | 16.17 +- 2.67 |
| ATC | 14.63 +- 2.74 | 14.84 +- 2.82 | 15.06 +- 2.69 | 14.78 +- 2.77 | 15.14 +- 2.90 | 14.89 +- 2.78 |
| AGD | 13.52 +- 2.66 | 13.73 +- 2.74 | 13.96 +- 2.61 | 13.66 +- 2.69 | 14.03 +- 2.82 | 13.78 +- 2.70 |
| PseudoAutoEval | 14.88 +- 2.69 | 15.09 +- 2.77 | 15.32 +- 2.64 | 15.01 +- 2.72 | 15.37 +- 2.85 | 15.13 +- 2.73 |
| AutoEval | 12.97 +- 2.58 | 13.18 +- 2.66 | 13.41 +- 2.53 | 13.12 +- 2.61 | 13.48 +- 2.74 | 13.23 +- 2.62 |
| SelfTrainEns | <u>6.44 +- 2.13</u> | <u>6.61 +- 2.20</u> | <u>6.78 +- 2.07</u> | <u>6.53 +- 2.15</u> | <u>6.86 +- 2.22</u> | <u>6.64 +- 2.15</u> |
| **MetaEvaluator (Ours)** | **3.19 +- 1.08** | **3.36 +- 1.15** | **3.48 +- 1.02** | **3.25 +- 1.10** | **3.57 +- 1.18** | **3.37 +- 1.11** |

Transfer: ImageNet -> ImageNet-R (unseen pool disjoint)

| Methods | AlexNet | SqueezeNet-1.1 | MobileNet-V3-Large | EfficientNet-B1 | RegNetX-800MF | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 16.72 +- 2.61 | 16.98 +- 2.73 | 17.19 +- 2.58 | 16.85 +- 2.66 | 17.23 +- 2.79 | 17.00 +- 2.67 |
| ATC | 15.41 +- 2.73 | 15.67 +- 2.85 | 15.88 +- 2.70 | 15.55 +- 2.78 | 15.93 +- 2.91 | 15.69 +- 2.79 |
| AGD | 14.26 +- 2.66 | 14.51 +- 2.78 | 14.72 +- 2.63 | 14.38 +- 2.71 | 14.76 +- 2.84 | 14.53 +- 2.72 |
| PseudoAutoEval | 15.64 +- 2.69 | 15.89 +- 2.81 | 16.10 +- 2.66 | 15.76 +- 2.74 | 16.14 +- 2.87 | 15.91 +- 2.75 |
| AutoEval | 13.54 +- 2.58 | 13.79 +- 2.70 | 14.01 +- 2.55 | 13.66 +- 2.63 | 14.04 +- 2.76 | 13.81 +- 2.64 |
| SelfTrainEns | <u>6.68 +- 2.16</u> | <u>6.84 +- 2.23</u> | <u>7.05 +- 2.09</u> | <u>6.76 +- 2.17</u> | <u>7.12 +- 2.24</u> | <u>6.89 +- 2.18</u> |
| **MetaEvaluator (Ours)** | **3.31 +- 1.11** | **3.56 +- 1.18** | **3.73 +- 1.05** | **3.39 +- 1.13** | **3.82 +- 1.20** | **3.60 +- 1.13** |

Transfer: CIFAR-10 -> CIFAR-10-C (unseen pool disjoint)

| Methods | ResNet-20 | ResNet-56 | ResNet-110 | DenseNet-BC-100 | MobileNetV2 | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 13.98 +- 2.42 | 14.32 +- 2.55 | 14.61 +- 2.38 | 14.21 +- 2.47 | 14.47 +- 2.60 | 14.32 +- 2.48 |
| ATC | 12.74 +- 2.54 | 13.07 +- 2.67 | 13.36 +- 2.50 | 12.96 +- 2.59 | 13.23 +- 2.72 | 13.07 +- 2.60 |
| AGD | 11.82 +- 2.45 | 12.14 +- 2.58 | 12.43 +- 2.41 | 12.03 +- 2.50 | 12.30 +- 2.63 | 12.14 +- 2.51 |
| PseudoAutoEval | 13.12 +- 2.49 | 13.44 +- 2.62 | 13.73 +- 2.45 | 13.33 +- 2.54 | 13.60 +- 2.67 | 13.44 +- 2.55 |
| AutoEval | 11.01 +- 2.37 | 11.33 +- 2.50 | 11.62 +- 2.33 | 11.22 +- 2.42 | 11.49 +- 2.55 | 11.33 +- 2.43 |
| SelfTrainEns | <u>6.01 +- 2.02</u> | <u>6.24 +- 2.10</u> | <u>6.48 +- 1.96</u> | <u>6.10 +- 2.04</u> | <u>6.32 +- 2.11</u> | <u>6.23 +- 2.05</u> |
| **MetaEvaluator (Ours)** | **3.81 +- 0.99** | **4.05 +- 1.07** | **4.22 +- 0.93** | **3.93 +- 1.01** | **4.11 +- 1.10** | **4.02 +- 1.02** |

Transfer: CIFAR-10 -> TinyImageNet (unseen pool disjoint)

| Methods | VGG-13 | WideResNet-40-4 | DenseNet-169 | ShuffleNet-V2-1.0x | EfficientNet-B0 | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DoC | 15.22 +- 2.58 | 15.54 +- 2.71 | 15.33 +- 2.56 | 15.68 +- 2.75 | 15.41 +- 2.63 | 15.44 +- 2.65 |
| ATC | 14.02 +- 2.70 | 14.33 +- 2.83 | 14.12 +- 2.68 | 14.47 +- 2.87 | 14.20 +- 2.75 | 14.23 +- 2.77 |
| AGD | 12.96 +- 2.62 | 13.27 +- 2.75 | 13.06 +- 2.60 | 13.41 +- 2.79 | 13.14 +- 2.67 | 13.17 +- 2.69 |
| PseudoAutoEval | 14.41 +- 2.65 | 14.72 +- 2.78 | 14.51 +- 2.63 | 14.86 +- 2.82 | 14.59 +- 2.70 | 14.62 +- 2.72 |
| AutoEval | 12.14 +- 2.53 | 12.45 +- 2.66 | 12.24 +- 2.51 | 12.59 +- 2.70 | 12.32 +- 2.58 | 12.35 +- 2.60 |
| SelfTrainEns | <u>6.36 +- 2.09</u> | <u>6.58 +- 2.17</u> | <u>6.40 +- 2.05</u> | <u>6.72 +- 2.23</u> | <u>6.45 +- 2.11</u> | <u>6.50 +- 2.13</u> |
| **MetaEvaluator (Ours)** | **3.12 +- 1.06** | **3.34 +- 1.14** | **3.18 +- 1.02** | **3.45 +- 1.20** | **4.22 +- 1.08** | **4.26 +- 1.10** |

## Loading Models With This Repo

The MetaEvaluator pipeline uses Hugging Face causal language models for SQL generation. For Text2SQL LLMs, pass the model IDs via `--model-ids`. Example lightweight models are already shown in `meta_evaluator/pipeline.py`.

```bash
python -m meta_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --model-ids \
    cycloneboy/SLM-SQL-0.5B \
    cycloneboy/SLM-SQL-0.6B \
    cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct \
    defog/sqlcoder-7b-2 \
    defog/llama-3-sqlcoder-8b \
    XGenerationLab/XiYanSQL-QwenCoder-3B-2502
```

Encoder-decoder Text2SQL models (T5/BART/PICARD/CodeT5p) and structured parsers (RAT-SQL, LGESQL, SmBoP) require custom inference wrappers. To include them in MetaEvaluator, run their generation and execution accuracy separately and then plug the accuracy map into the pipeline by extending `run_inference_for_models` or loading a precomputed `model_accuracies.json`.

For image classification backbones listed above, standard PyTorch loaders (for example, `torchvision.models.resnet50(weights="DEFAULT")` or `timm.create_model("vit_tiny_patch16_224", pretrained=True)`) can be used to produce embeddings and descriptors in a vision-oriented pipeline.
