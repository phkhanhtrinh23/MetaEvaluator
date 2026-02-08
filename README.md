# MetaEvaluator

<!-- > The rapid advancement of machine learning has led to an unprecedented expansion of model ecosystems, making it increasingly difficult to assess the reliability of newly emerging models on unseen and unlabeled data. Existing evaluation pipelines typically rely on costly annotation, repeated fine-tuning, or assumptions that do not generalize well to new models. MetaEvaluator is a cost-effective, model-agnostic framework for fast, label-free evaluation of unseen models across diverse architectures and modalities. MetaEvaluator applies meta-learning over a pool of reference models, acquiring an effective initialization for accurate assessment of unseen models, thereby amortizing evaluation cost and eliminating the need for per-model retraining. To the best of our knowledge, this is the first model-agnostic framework that enables the evaluation of new models on unlabeled datasets. Extensive experiments demonstrate that MetaEvaluator delivers stable and accurate performance estimates at substantially lower cost than conventional approaches, enabling scalable benchmarking on unlabeled datasets for emerging models. -->

## Overview

![MetaEvaluator Architecture](./resources/training_pipeline.png)

## Components

- **Shift Descriptor Pipeline** (`shift_descriptor/`): builds distribution shift descriptors from train/test embeddings.
- **MetaEvaluator** (`meta_evaluator/`): meta-learns a predictor of model accuracy from descriptor pairs.

## Requirements

```bash
pip install torch transformers scipy scikit-learn matplotlib tqdm numpy peft
```

> Some models listed below may require gated access on Hugging Face. Use `--model-ids alias=model_id` to point to approved checkpoints once access is granted.

## Shift Descriptor Pipeline

This pipeline extracts pooled embeddings from lightweight Hugging Face LLMs on the provided Text2SQL SFT data, computes the three descriptor metrics (Frechet, Mahalanobis, and Sliced Wasserstein distances), stores the resulting matrices, projects them with PCA, and reports pairwise similarities between the descriptor vectors.

### Example

```bash
python -m shift_descriptor.pipeline \
  --train-path <train_filename>.json \
  --test-path <test_filename>.json \
  --output-dir outputs
```

Key arguments:

- `--model-ids`: Optional list overriding the three default lightweight models (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `HuggingFaceH4/zephyr-3b-beta`, `microsoft/Phi-3-mini-4k-instruct`). Use `alias=model_id` to keep filenames readable.
- Suffix an entry with `:remote` (e.g., `mamba=state-spaces/mamba-2.8b-slimpj:remote`) to auto-enable `trust_remote_code`. This is required for architectures like Mamba, RWKV, MiniCPM, and InternLM2.
- `--lora-r`: LoRA rank applied on-the-fly to every loaded model (default 8). Set to `0` to disable if you prefer the raw weights.
- `--metric-max-points`: subsample cap (default 4096) for metric computation to avoid loading very large embedding matrices fully in RAM.
- `--prompt-template`: Path to the Text-to-SQL prompt template (defaults to `prompts/text2sql_prompt.tmpl`). The template injects database ID, schema tables, foreign keys, and question/context into the instruction block before embedding.
- `--context-fields`: Which JSON fields feed the "Additional context" slot (default: `evidence matched_contents text`).
- `--use-plain-text`: Skip the template entirely and embed the raw `--text-field` value.
- `--device`: Force computation on `cuda`, `cpu`, or `mps`.
- `--lora-r`: Rank for on-the-fly LoRA adapters (default 8, set to 0 to disable).
- `--metric-max-points`: Optional subsample cap for descriptor metrics (0 keeps every embedding row).
- `--cka-max-points`: Cap (default 4096) on normalized samples used when measuring representational similarity (CKA/Procrustes).
- Embedding `.npz` files are cached in `outputs/`. If a model/split pair already exists, the pipeline loads it instead of recomputing.
- `--skip-embedding-plots`, `--scatter-max-points`, `--scatter-seed`: control the per-model embedding scatter figure (each subplot runs its own PCA over that model's train/test embeddings).
- `--num-projections`: Controls the Monte-Carlo estimate of the sliced Wasserstein distance.
- The pipeline skips models that fail to produce valid embeddings (e.g., due to missing deps or NaN outputs), deletes their partial `.npz` files, and records the error message under `failed_models` in the summary.

### Outputs

All artifacts land in `outputs/` by default:

- `*_train_embeddings.npz` / `*_test_embeddings.npz`: pooled embeddings for each model and split.
- `shift_descriptor_matrix.npy`: rows -> models, cols -> descriptor metrics (Frechet, Mahalanobis, Sliced Wasserstein, mean-shift norm, train anisotropy).
- `shift_descriptor_pca.png`: 2D PCA projection for visual triage.
- `all_models_embedding_scatter.png`: combined PCA overlay where each model/split is color/marker coded; embeddings are normalized per model before PCA.
- `per_model_embedding_scatter.png`: tiled scatter figure where each subplot shows a model-specific PCA projection of its train vs. test splits (also using normalized embeddings).
- `pairwise_similarity.json`: cosine similarity between each pair of descriptor vectors.
- `cka_similarity.json`: pairwise CKA scores between normalized embedding spaces.
- `architecture_alignment.json`: aggregates of within/between architecture-family CKA scores to show how well clusters align with expected signatures.
- `cka_heatmap.png`: visual heatmap of the pairwise CKA matrix.
- `architecture_alignment.png`: bar-chart view of within/between-family mean CKA values.
- `diagnostics/`: normalization sweeps (overlays, per-model scatters), cosine histograms, Procrustes residuals, and a mean-shift/anisotropy bar chart for quick inspection.
- `shift_descriptor_summary.json`: consolidated metadata with file pointers, metric names, and sample counts.
- When templating is active, the summary file also records the prompt template path that was used.

### Extra requirements for specialized backbones

Some architectures rely on custom kernels. Install the corresponding extras before running the pipeline with those checkpoints:

- **Mamba** (`state-spaces/mamba-*`): `pip install mamba-ssm`
- **RWKV7** (`RWKV/RWKV7-*`): `pip install fla`
- **RecurrentGemma** (`google/recurrentgemma-*`): accept the model's license on Hugging Face and install `pip install recurrentgemma`
- **LoRA** (all models when enabled): `pip install peft`

The `EmbeddingExtractor` now falls back to recurrent `state` tensors when `last_hidden_state` is unavailable, so these models integrate seamlessly once their deps are available.

## MetaEvaluator (Text2SQL Meta-Learning)

The MetaEvaluator pipeline predicts execution accuracy for new Text-to-SQL models using shift descriptors and a meta-learned context adaptation strategy.

### Usage

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
- Splits the training JSON into `meta_train`/`meta_val`/`meta_test` (default 60/20/20), renders prompts, and caches embeddings per model/split (FP16 on CUDA, optional LoRA).
- Generates SQL for each split and computes exact/execution accuracy against SQLite DBs under `data/database/<db_id>/<db_id>.sqlite` (or `db_path` fallback). Prediction/accuracy JSON is cached and reused if sample counts match.
- Builds shift descriptors (Frechet/Mahalanobis/SWD) between meta_train and other splits, normalizes them, and meta-trains a three-layer MLP (MetaEvaluator) with early stopping and optional meta regularizers (`meta_reg_lambda`, `meta_reg_beta`).
- Held-out models (`--test-model-ids`) are evaluated by adapting the context from their support descriptors and then predicting on meta-test/real-test splits.
- Outputs live under `outputs/meta_evaluator/` (metrics, predictions, checkpoints). A helper `scripts/inspect_meta_evaluator.py` prints the saved checkpoint structure, and `scripts/rename_model_outputs.py` migrates old tail-based filenames to full model ids.

### Text2SQL Model Pools

Text2SQL model pool: **78 total**.

- Structured Text2SQL Parsers (8): RAT-SQL; LGESQL; SmBoP; RESDSQL; Clause-SmBoP; IRNet; BRIDGE; ValueNet / RYANSQL.
- Encoder--Decoder Models (10): PICARD T5-large (Spider); PICARD T5-base (Spider); T5-small Text2SQL; FLAN-T5-base Text2SQL (QLoRA finetune); Priyanshu05/text-to-sql T5; Shubh7/T5-Small Text2SQL; BART-large NL2SQL (LarkAI); BART-large NL2SQL (SwastikM); CodeT5p-770M NL2SQL; FronyAI/natural2sql-ko.
- SQLCoder / SLM-SQL / CscSQL / Hrida Families (24): SQLCoder-7B-2; SQLCoder-15B; SQLCoder2-15B; SQLCoder-70B-alpha; Llama-3 SQLCoder-8B; SQLCoder-7B GGUF; SQLCoder-7B GPTQ; SQLCoder-7B AWQ; SLM-SQL-0.5B; SLM-SQL-0.6B; SLM-SQL-1.3B; SLM-SQL-1.5B; SLM-SQL-Base-0.5B; SLM-SQL-Base-0.6B; SLM-SQL-Base-1B; SLM-SQL-Base-1.3B; CscSQL-Merge Qwen2.5-Coder-0.5B; CscSQL-Merge Qwen2.5-Coder-1.5B; CscSQL-Merge Qwen2.5-Coder-3B; CscSQL-Merge Qwen2.5-Coder-7B; CscSQL-Grpo Qwen2.5-Coder-3B; CscSQL-Grpo Qwen2.5-Coder-7B; Hrida-T2SQL-3B v0.1; Hrida-T2SQL-3B v0.2.
- Other LLMs (10): DeepSeek-Coder-1.3B; Snowflake Arctic-Text2SQL-R1-7B; DeepSeek-R1-Distill-Qwen-1.5B; WizardCoder Spider-NatSQL; Mistral-7B; Llama-3.1-8B Instruct; Qwen2.5-7B Instruct; Qwen2.5-0.5B Instruct; Mistral-7B Instruct v0.3; DeepSeek-Coder-6.7B Instruct.
- Modern General LLM Backbones and Hybrids (26): DeepSeek-V3 Base; Open-R1 reasoning model; OLMo-2 7B Base; OLMo-2 13B Instruct; gemma-3-12b-it; gemma-3-27b-it; Mistral-Small-3.1 Base; Mistral-Small-3.1 Instruct; Llama-4-Scout; Llama-4-Maverick; Qwen3-4B Instruct; Qwen3-14B Instruct; Qwen3-32B Base; SmolLM3-3B Base; SmolLM3-3B Instruct; Kimi-K2 Instruct; Kimi-K2 Thinking; GPT-OSS-21B MoE; GPT-OSS-117B MoE; GLM-4.5 Chat; GLM-4.6 Chat; MiniMax-M1; MiniMax-M2; RWKV-4-430M; RWKV-7B; Mamba2-1.3B.

Unseen Text2SQL test pool: **5 total**: Meta-Llama-3-70B; Qwen2.5-32B; XiYanSQL-14B; Ministral-3-14B; gemma-2-2b.

### Image Classification Model Pools

Image classification model pool: **43 total**.

- Classic CNN Baselines (2): LeNet-5; AlexNet.
- VGG Family (4): VGG-11; VGG-13; VGG-16; VGG-19.
- Residual-Style Networks (5): ResNet-18; ResNet-34; ResNet-50; ResNet-101; ResNet-152.
- Wide Residual Networks (4): WideResNet-16-8; WideResNet-28-10; WideResNet-40-2; WideResNet-40-4.
- Dense Connectivity (4): DenseNet-121; DenseNet-161; DenseNet-169; DenseNet-201.
- Efficient Mobile CNNs (10): MobileNet-V1; MobileNet-V2; MobileNet-V3-Small; MobileNet-V3-Large; ShuffleNet-V2-0.5x; ShuffleNet-V2-1.0x; ShuffleNet-V2-1.5x; EfficientNet-B0; EfficientNet-B1; EfficientNet-B2.
- Scaled / Lightweight CNNs (5): SqueezeNet-1.0; SqueezeNet-1.1; RegNetX-400MF; RegNetX-800MF; RegNetY-400MF.
- CIFAR-Standard Robust Baselines (9): ResNet-20 (CIFAR); ResNet-56 (CIFAR); ResNet-110 (CIFAR); PreAct-ResNet-18; PyramidNet-110; Shake-Shake-26-2x32d; ResNeXt-29-8x64d (CIFAR); DenseNet-BC-100 (CIFAR); MobileNetV2 (CIFAR).

Unseen image test pool: **5 total**: ResNeXt-50-32x4d; RegNetY-8GF; ConvNeXt-Tiny; ViT-Tiny; DeiT-Small.

## Loading Models With This Repo

The MetaEvaluator pipeline uses Hugging Face **causal language models** for SQL generation. For Text2SQL LLMs, pass the model IDs via `--model-ids`. Example lightweight models are already shown in `meta_evaluator/pipeline.py`.

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

Model ID examples discovered on Hugging Face (non-exhaustive, for the Text2SQL LLM families):

- SQLCoder family: `defog/sqlcoder-7b-2`, `defog/sqlcoder`, `defog/sqlcoder2`, `defog/llama-3-sqlcoder-8b` (plus GGUF/GPTQ/AWQ variants under `TheBloke/*` or community quant repos).
- SLM-SQL family: `cycloneboy/SLM-SQL-0.5B`, `cycloneboy/SLM-SQL-0.6B`, `cycloneboy/SLM-SQL-1.3B`, `cycloneboy/SLM-SQL-1.5B`, and base variants like `cycloneboy/SLM-SQL-Base-0.5B`.
- CscSQL family: `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct`, `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct`, `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-3B-Instruct`, `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct`, `cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct`, `cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct`.
- Hrida-T2SQL family: `HridaAI/Hrida-T2SQL-3B-V0.1`, `HridaAI/Hrida-T2SQL-3B-V0.2` (quantized GGUF variants under `mradermacher/*`).
- Other LLMs: `deepseek-ai/deepseek-coder-1.3b-base`, `deepseek-ai/deepseek-coder-6.7b-instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Snowflake/Arctic-Text2SQL-R1-7B`, `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`.

Encoder--decoder Text2SQL models (T5/BART/PICARD/CodeT5p) and structured parsers (RAT-SQL, LGESQL, SmBoP, etc.) require custom inference wrappers. To include them in MetaEvaluator, run their generation/exec accuracy separately and then plug the accuracy map into the pipeline by extending `run_inference_for_models` or loading a precomputed `model_accuracies.json`.

For image classification backbones listed above, standard PyTorch loaders (e.g., `torchvision.models.resnet50(weights="DEFAULT")` or `timm.create_model("vit_tiny_patch16_224", pretrained=True)`) can be used to produce embeddings/descriptors in a vision-oriented pipeline.
