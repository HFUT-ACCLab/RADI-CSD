# Configuration and execution

All examples below refer to paths from the repository root, except the launch commands explicitly run inside `run/`. Config values containing spaces, such as `Apollo Go`, should be quoted.

## Path convention

Use absolute paths for model, data, teacher-cache, training-output, and prediction-output locations. This avoids ambiguity because the current `train_eval.sh` and prediction wrappers must run from `run/`.

For example, if the repository is `/home/user/RADI`, a Biden training YAML should use:

```yaml
model_name_or_path: /models/Qwen3-8B
dataset: Biden_train
eval_dataset: Biden_eval
dataset_dir: /home/user/RADI/datasets/MT-CSD/sft_addlabel_noInstance_CUSTOM-LOSS_instruct/Biden
kd_dir: /home/user/RADI/datasets/MT-CSD/teacher_extract_tagged/Biden/teacher_samples
output_dir: /home/user/RADI/result_save/MT-CSD/Biden/radi_seed42
```

Keep the remaining experiment hyperparameters consistent with the intended run.

In the corresponding prediction wrapper:

```python
root_dir = "/home/user/RADI"
dataset_dir = root_dir + "/datasets/MT-CSD/sft_addlabel_noInstance_CUSTOM-LOSS_instruct/Biden"
check_points_dir = root_dir + "/result_save/MT-CSD/Biden/radi_seed42"
output_dir = root_dir + "/result_save/predict/MT-CSD/Biden/radi_seed42"
base_config_path = "MT-CSD_Biden_predict_qwen.yaml"
```

The prediction YAML must use the same student base model as training. Its data and output paths are overridden by the wrapper, but the YAML must still be syntactically valid before those overrides can happen.

## Corrections required in the supplied archive

1. **Prediction YAML syntax.** Fix missing spaces after mapping colons. In `run/MT-CSD_Biden_predict_qwen.yaml`, change `output_dir:path/to/result_save/` into `output_dir: path/to/result_save/`. In `run/C-MTCSD_Apollo_Go_predict.yaml`, fix `dataset_dir:path/to/datasets/` and inspect the remaining mappings. Replace placeholder values with the intended locations.
2. **MT-CSD data directory names.** The supplied directory is `sft_addlabel_noInstance_CUSTOM-LOSS_instruct`. Several training YAMLs and prediction wrappers still refer to `sft_addlabel_noInstance_CUSTOM-LOSS_instruct_prompt_1_0428`; update them to the included directory.
3. **Working directory.** `./datasets` inside a training configuration is resolved from the process working directory, not automatically from the repository root. With `cd run`, use absolute paths or `../datasets/...`.
4. **SpaceX teacher cache.** Its training YAML currently points to `Tesla_2/teacher_samples`. Replace it with the cache generated from SpaceX training samples. Matching numeric IDs across targets does not make the caches interchangeable.
5. **Teacher output/cache paths.** Teacher scripts write under `teacher_extract_tagged`, while training YAMLs still reference `teacher_extract_two/*_2`. Set `kd_dir` to the actual `OUTPUT_DIR/teacher_samples` of the same target.
6. **Chinese prompt JSON.** Replace the example's full-width punctuation with standard JSON syntax:

```json
{
  "Reasoning": "<STEP_1>...<STEP_END><STEP_2>...<STEP_END><STEP_3>...<STEP_END>",
  "Stance": "favor"
}
```

The label rules should consistently refer to `favor`, `against`, and `none`.

## Teacher extraction settings

Edit `MODEL_PATH`, `DATA_PATH`, and `OUTPUT_DIR` in each extraction script. The packaged defaults are `EXTRACT_MODE="two_stage"`, `TOP_K=50`, `MAX_NEW_TOKENS=512`, and `enable_thinking=False`. The English/Chinese structured prompt still requests three explicit reasoning stages. Use the training split for teacher supervision.

The extractor saves `.pt` files that pass its current marker and label-position checks. It does not filter by gold-label correctness when saving; the trainer skips incorrect teacher labels for feature and logit distillation. The extractor's structure checks do not yet fully validate JSON or every step tag.

Existing `<id>.pt` files are skipped. Use a new output directory after changing the input data, prompt, teacher model, or extraction settings. Keep sample IDs unchanged and separate caches by target.

## Launcher behavior

The current launcher uses background commands followed by `wait`, so its jobs are sequential. It does not reliably stop the full pipeline after every child-process failure. For a failure-stopping launcher, place the following at the top and remove each trailing `&` and the associated standalone `wait`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
```

This documentation does not modify the supplied launcher or configurations.

## Verification before a full run

1. Parse every YAML and confirm all configured files exist.
2. Confirm `import llamafactory` resolves to the bundled checkout.
3. Generate and inspect a few teacher outputs: correct input, three stage boundaries, stance label, and four saved vectors.
4. Run one target's training and prediction before launching all ten.
5. Select checkpoints using the development split and record seeds, configuration, package versions, hardware, and dataset/cache identifiers.
6. Recompute the metric required by the paper from saved predictions; the currently named `F1` is the favor/against mean, not three-class macro-F1.
