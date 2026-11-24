# RADI

Reasoning Anchor guided implicit CoT DIstillation for conversational stance detection.

RADI uses a structured teacher prompt to analyze a conversation in three stages. Hidden states at the stage boundaries and the answer predictor position supervise a student through four projection heads. Training combines feature distillation, answer-logit distillation, and gold-label supervision. The student performs stance detection without generating explicit rationales at inference.

## Scope

This repository contains the target-specific training and evaluation pipeline for MT-CSD and C-MTCSD: processed train/dev/test splits, English and Chinese teacher extraction scripts, a modified LLaMA-Factory checkout, and training/prediction configurations for ten targets. The main entry point is `run/train_eval.sh`.

**Release status:** the supplied configurations still require path corrections and two prediction YAML syntax fixes. Complete [configuration setup](docs/CONFIGURATION.md) before running. Teacher caches and model weights are not bundled. This package does not provide launchers for all paper analyses, such as cross-target evaluation, inference efficiency, or representation visualization.

## Repository structure

```text
.
├── LLaMA-Factory/          # Modified student training/evaluation framework
├── datasets/              # Processed MT-CSD and C-MTCSD splits and registrations
├── teacher_anchor_extract/
│   ├── extract_radi_teacher_en.py
│   └── extract_radi_teacher_cn.py
├── run/                   # Training YAMLs, prediction YAMLs and wrappers
│   └── train_eval.sh
├── requirements.txt       # Original mixed Conda/pip environment inventory
└── docs/
    ├── DATA.md
    └── CONFIGURATION.md
```

## 1. Environment

Use Linux, Bash, and a CUDA-enabled PyTorch environment. The configured models are Qwen3-32B (teacher) and Qwen3-8B (student). Teacher extraction uses BF16 and `device_map="auto"`; GPU allocation and memory availability must accommodate the model and sequence lengths.

The root `requirements.txt` is a Conda-style inventory containing `pypi_0` entries, **not a pip-installable requirements file**. It records Python 3.12.13, PyTorch 2.2.2+cu118, Transformers 4.55.0, PEFT 0.15.2, Accelerate 1.7.0, and Datasets 3.6.0. These recorded versions are not a validated fresh-install lockfile.

In an existing working experiment environment, install the bundled modified framework from the repository root:

```bash
python -m pip install -e ./LLaMA-Factory --no-deps
python -c "import llamafactory; print(llamafactory.__file__)"
python -m pip check
```

The import path must point to this checkout. Dependencies include those in `LLaMA-Factory/requirements.txt`, PyTorch, scikit-learn, natsort, and the metric dependencies specified in `LLaMA-Factory/setup.py`. Resolve dependency conflicts before running; a fresh-install environment must be validated separately.

## 2. Prepare data and teacher supervision

Processed splits are included. See [data format and split sizes](docs/DATA.md).

In each teacher extraction script, set `MODEL_PATH`, `DATA_PATH`, and `OUTPUT_DIR`. Use the English script for MT-CSD and the Chinese script for C-MTCSD. Each invocation processes one target's training split. Correct the Chinese JSON example as described in [configuration setup](docs/CONFIGURATION.md).

```bash
# Run from the repository root; choose available GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 python teacher_anchor_extract/extract_radi_teacher_en.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python teacher_anchor_extract/extract_radi_teacher_cn.py
```

Generate caches for every target to be trained, then point each training YAML's `kd_dir` to that target's `teacher_samples/` directory. Student training reads these offline caches and does not load the teacher model.

## 3. Configure training and prediction

Follow [configuration setup](docs/CONFIGURATION.md). In particular:

- Set student model paths in both training and prediction YAMLs.
- Point `dataset_dir` to the included target directory, which contains `dataset_info.json`.
- Set `kd_dir` to the corresponding teacher cache.
- Make the prediction wrapper's `check_points_dir` match the training `output_dir`.
- Resolve paths relative to the `run/` working directory, or use absolute paths.

## 4. Run the main experiment

After the preparation above:

```bash
cd run
CUDA_VISIBLE_DEVICES=0 bash train_eval.sh
```

The script runs each training command followed by its prediction wrapper, sequentially, across five MT-CSD targets and five C-MTCSD targets. It does not extract teacher anchors.

For an initial single-target run, execute from `run/`:

```bash
llamafactory-cli train MT-CSD_Biden_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml
python run_multi_predict_Biden_noInstance_Multi-layers-topKL_prompt.py
```

The prediction wrappers enumerate saved `checkpoint*` directories. They do not independently select the best checkpoint; use development-set selection for the reported test result. Do not select a checkpoint by its test score.

## 5. Outputs and metrics

- Training checkpoints and trainer state: the training YAML's `output_dir`.
- Console logs: the `.log` files specified in `run/train_eval.sh` (appended with `>>`).
- Predictions: `generated_predictions.jsonl` in each prediction output directory, with `prompt`, `predict`, and `label` fields.
- Metric artifacts: `predict_results.json` and `generated_predictions_f1.jsonl`. The latter contains a JSON score line followed by a text classification report; it is not a uniform JSONL dataset.

The current metric implementation reports three-class accuracy and the mean of the favor and against F1 scores. The `none` class is excluded from the F1 average, but its samples still affect false positives/negatives for the other classes. Trainer metrics are on a 0–1 scale, whereas `report_score` exports accuracy and favor/against mean F1 multiplied by 100. Its classification report includes per-class metrics and a separate macro average; these quantities should not be interchanged.

The release does not include a script aggregating all targets into the paper's dataset-level Overall metric. Use the aggregation rule defined in the paper when preparing tables.

## Implementation

Core RADI changes are located in:

| Path under `LLaMA-Factory/src/llamafactory/` | Purpose |
|---|---|
| `train/sft/trainer.py` | Teacher cache loading, layer/anchor alignment and joint training losses |
| `model/loader.py` | Four trainable projection heads |
| `hparams/finetuning_args.py` | Teacher cache configuration (`kd_dir`) |
| `data/converter.py`, `data/processor/supervised.py` | Preserve sample IDs and pass `kd_id` |
| `train/sft/metric.py`, `train/trainer_utils.py` | Stance metrics and prediction reports |
| `train/sft/workflow.py` | Training/evaluation integration and prediction export |

This is a functional map, not an exhaustive diff against upstream. Projection output size is currently fixed at 5120 in `model/loader.py`; check this when changing the teacher backbone.

## Acknowledgments and attribution

This implementation builds on LLaMA-Factory. Preserve its license and copyright notices in `LLaMA-Factory/`. Cite the original MT-CSD and C-MTCSD dataset papers when using their data. Code licensing does not replace the datasets' original usage conditions. Paper citation metadata and a code-specific license for the RADI additions should be supplied by the authors before a formal release.
