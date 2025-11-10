## Shift Descriptor Pipeline

This repo now includes a shift-descriptor workflow inspired by *main_modeval_icde.pdf*.  
The pipeline extracts pooled embeddings from lightweight Hugging Face LLMs on the provided Text2SQL SFT data, computes the three descriptor metrics (Fréchet, Mahalanobis, and Sliced Wasserstein distances), stores the resulting matrices, projects them with PCA, and reports pairwise similarities between the descriptor vectors.

### Requirements

```bash
pip install torch transformers scipy scikit-learn matplotlib tqdm numpy peft
```

> ⚠️  Some models listed in the accompanying figure (e.g., Llama 3.2 1B, Qwen3 4B, SmoLM3 3B) may require gated access on Hugging Face. Use `--model-ids alias=model_id` to point to approved checkpoints once access is granted.

### Usage

```bash
python -m shift_descriptor.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --test-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs
```

Key arguments:

- `--model-ids`: Optional list overriding the three default lightweight models (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `HuggingFaceH4/zephyr-3b-beta`, `microsoft/Phi-3-mini-4k-instruct`). Use `alias=model_id` to keep filenames readable.
- Suffix an entry with `:remote` (e.g., `mamba=state-spaces/mamba-2.8b-slimpj:remote`) to auto-enable `trust_remote_code`. This is required for architectures like Mamba, RWKV, MiniCPM, and InternLM2.
- `--lora-r`: LoRA rank applied on-the-fly to every loaded model (default 8). Set to `0` to disable if you prefer the raw weights.
- `--metric-max-points`: subsample cap (default 4096) for metric computation to avoid loading very large embedding matrices fully in RAM.
- `--prompt-template`: Path to the Text-to-SQL prompt template (defaults to `prompts/text2sql_prompt.tmpl`). The template injects database ID, schema tables, foreign keys, and question/context into the instruction block before embedding.
- `--context-fields`: Which JSON fields feed the “Additional context” slot (default: `evidence matched_contents text`).
- `--use-plain-text`: Skip the template entirely and embed the raw `--text-field` value.
- `--device`: Force computation on `cuda`, `cpu`, or `mps`.
- Embedding `.npz` files are cached in `outputs/`. If a model/split pair already exists, the pipeline loads it instead of recomputing.
- `--skip-embedding-plots`, `--scatter-max-points`, `--scatter-seed`: control the per-model embedding scatter figure (each subplot runs its own PCA over that model's train/test embeddings).
- `--num-projections`: Controls the Monte-Carlo estimate of the sliced Wasserstein distance.
- The pipeline skips models that fail to produce valid embeddings (e.g., due to missing deps or NaN outputs), deletes their partial `.npz` files, and records the error message under `failed_models` in the summary.

### Outputs

All artifacts land in `outputs/` by default:

- `*_train_embeddings.npz` / `*_test_embeddings.npz`: pooled embeddings for each model and split.
- `shift_descriptor_matrix.npy`: rows → models, cols → descriptor metrics.
- `shift_descriptor_pca.png`: 2D PCA projection for visual triage.
- `all_models_embedding_scatter.png`: combined PCA overlay where each model/split is color/marker coded (like the example you referenced).
- `per_model_embedding_scatter.png`: tiled scatter figure where each subplot shows a model-specific PCA projection of its train vs. test embeddings.
- `pairwise_similarity.json`: cosine similarity between each pair of descriptor vectors (capturing the “distribution similarity between each 2 out of 3 matrices” requirement).
- `shift_descriptor_summary.json`: consolidated metadata with file pointers, metric names, and sample counts.
- When templating is active, the summary file also records the prompt template path that was used.
- `per_model_embedding_scatter.png`: tiled scatter figure where each subplot shows a model-specific PCA projection of its train vs. test embeddings.

### Extra requirements for specialized backbones

Some architectures rely on custom kernels. Install the corresponding extras before running the pipeline with those checkpoints:

- **Mamba** (`state-spaces/mamba-*`): `pip install mamba-ssm`
- **RWKV7** (`RWKV/RWKV7-*`): `pip install fla`
- **RecurrentGemma** (`google/recurrentgemma-*`): accept the model’s license on Hugging Face and install `pip install recurrentgemma`
- **LoRA** (all models when enabled): `pip install peft`

The `EmbeddingExtractor` now falls back to recurrent `state` tensors when `last_hidden_state` is unavailable, so these models integrate seamlessly once their deps are available.
