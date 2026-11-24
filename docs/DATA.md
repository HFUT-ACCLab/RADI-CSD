# Data and teacher supervision

## Included data

The archive includes train/dev/test JSON arrays and `dataset_info.json` for five MT-CSD targets and five C-MTCSD targets. Preserve the supplied split membership and IDs when reproducing experiments. The following counts are measured from this archive, not inferred from dataset publications.

| Dataset | Target | Train | Dev | Test |
|---|---|---:|---:|---:|
| C-MTCSD | Apollo Go | 4,850 | 1,037 | 981 |
| C-MTCSD | Naked Resignation | 2,223 | 473 | 436 |
| C-MTCSD | Non-marriage Doctrine | 3,572 | 754 | 716 |
| C-MTCSD | Pre-made Meals | 3,631 | 776 | 789 |
| C-MTCSD | iPhone 15 | 2,826 | 624 | 576 |
| MT-CSD | Biden | 2,017 | 417 | 411 |
| MT-CSD | Bitcoin | 2,283 | 431 | 538 |
| MT-CSD | SpaceX | 1,440 | 268 | 321 |
| MT-CSD | Tesla | 2,552 | 536 | 540 |
| MT-CSD | Trump | 2,633 | 557 | 561 |

## Record format

Each file is a JSON array. This synthetic example illustrates the schema:

```json
[
  {
    "id": 0,
    "instruction": "Determine the stance toward the target. Output only favor, against, or none.",
    "input": "[Conversation History]\nNo history available.\n\n[Current Comment]\nI support this project.\n\n[Target]\nExample project\n\n[Label]\n",
    "output": "favor"
  }
]
```

C-MTCSD uses `[历史对话]`, `[当前评论]`, `[目标]`, and `[Label]`. The extractors support both sets of headings. Headers must appear once, in order, on their own lines. The final label section is empty; gold labels are stored in `output`.

For teacher extraction, the source `instruction` is replaced with the script's structured reasoning instruction. The comment, history and target are parsed from `input`. The gold label is not inserted into the teacher prompt.

`dataset_info.json` registers `<target>_train`, `<target>_eval`, and `<target>_test`. The `_eval` registration points to `full_content_dev.json`. Some registrations also contain historical entries; only use entries whose referenced files are included.

## Teacher cache

For each target, set the extractor's `OUTPUT_DIR`. It writes:

```text
OUTPUT_DIR/
├── teacher_samples/
│   ├── 0.pt
│   ├── 1.pt
│   └── ...
├── generated_texts.json
└── failed_samples.json
```

`teacher_samples/<id>.pt` contains:

| Field | Content |
|---|---|
| `doc_id` | Original sample ID as a string |
| `teacher_step_hidden` | Final-layer teacher vectors at the three stage-ending markers |
| `answer_predictor_hidden` | Final-layer state immediately before the generated stance-label token |
| `v`, `i` | Top-k raw answer logits and their vocabulary indices |
| `predicted_label` | Teacher stance label |
| `generated_text`, `generated_only_ids` | Generated response and token IDs |
| `step_positions`, `answer_token_pos`, `answer_token_id` | Positions/ID used during extraction |
| `extract_mode` | Extraction mode identifier |

The trainer joins student examples and teacher files by `kd_id`. IDs must match exactly; do not renumber records after cache generation. Numeric IDs may recur in different targets, so each target requires its own cache directory.

The teacher and student token-ID meanings must be compatible for vocabulary-index logit distillation. Recheck tokenizers before replacing the Qwen backbones.

Missing or teacher-incorrect samples do not contribute their corresponding distillation losses in the supplied trainer. Gold-label SFT supervision remains available. Report extracted/failed/retained counts rather than assuming every row supplies usable teacher supervision.

## Distribution

Teacher caches and model weights are not included; generate the caches using the provided scripts. There is no cache download URL in the supplied release. If downloadable caches are later provided, document their teacher version, prompts, extraction settings, sample counts, checksums and target mapping.

Processed dataset files retain the original datasets' usage conditions. Authors should document the original data sources, dataset citations, preprocessing provenance and redistribution terms before public release. The framework's code license does not establish permission to redistribute dataset text.
