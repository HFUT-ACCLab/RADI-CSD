"""RADI teacher extraction for tagged Alpaca data (id/instruction/input/output).

Edit MODEL_PATH, DATA_PATH and OUTPUT_DIR before running.
English and Chinese input section headers are supported.
This revision changes data parsing, preflight validation and ID handling.
The teacher prompt, thinking setting, extraction and loss-cache format are
preserved from the supplied script. Correct-answer filtering remains in the
RADI trainer; this script also saves structurally accepted wrong predictions.
Use a NEW OUTPUT_DIR when migrating from the old incompatible input parser.
An existing CUDA_VISIBLE_DEVICES environment variable is respected.
"""
import os
# [新增] 缓解 PyTorch 显存碎片化导致的 OOM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3") 
import gc
import numpy as np
import torch
import torch.nn.functional as F
import json
import traceback

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile
import re
from typing import Dict, List, Tuple
import ast

STANCE_LABELS = {"favor", "against", "none"}
STEP_END_MARKER = "<STEP_END>"

def parse_tagged_input(sample):
    """Read history/current comment/target from tagged Alpaca input.

    `instruction` is replaced by the teacher's structured prompt.
    `output` is a gold label and must never be inserted into that prompt.
    """
    if not isinstance(sample, dict):
        raise ValueError("Each dataset row must be an object.")
    text = sample.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing or empty string field: input")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    aliases = {
        "Conversation History": "history",
        "历史对话": "history",
        "Current Comment": "current",
        "当前评论": "current",
        "Target": "target",
        "目标": "target",
        "Label": "label",
        "标签": "label",
    }
    pattern = re.compile(
        r"^[ \t]*\[(" + "|".join(re.escape(k) for k in aliases)
        + r")\][ \t]*$", flags=re.MULTILINE
    )
    headers = list(pattern.finditer(text))
    actual = [aliases[m.group(1)] for m in headers]
    expected = ["history", "current", "target", "label"]
    if actual != expected:
        raise ValueError(
            "input must contain history, current comment, target and label "
            "headers exactly once in this order, each on its own line. "
            f"Detected headers: {[m.group(1) for m in headers]}"
        )
    if text[:headers[0].start()].strip():
        raise ValueError("Unexpected content before the history header.")
    parts = {}
    for i, header in enumerate(headers):
        stop = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        parts[aliases[header.group(1)]] = text[header.end():stop].strip()
    if not parts["current"]:
        raise ValueError("Current comment must not be empty.")
    if not parts["target"]:
        raise ValueError("Target must not be empty.")
    if parts["label"]:
        raise ValueError("The [Label] section must be empty; gold belongs in output.")
    history = parts["history"]
    if not history:
        history = ("没有历史对话。" if headers[0].group(1) == "历史对话"
                   else "No history available.")
    return history, parts["current"], parts["target"]


def load_and_validate_dataset(path):
    """Validate all rows before loading the teacher model."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            rows = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: line {error.lineno}, column {error.colno}. "
            "Provide a complete JSON array, including the closing ]."
        ) from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array.")
    seen = set()
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                raise ValueError("Expected a JSON object.")
            for field in ("id", "instruction", "input", "output"):
                if field not in row:
                    raise ValueError(f"Missing field: {field}")
            if not isinstance(row["instruction"], str):
                raise ValueError("instruction must be a string.")
            doc_id = get_doc_id(row, index)
            if doc_id in seen:
                raise ValueError(f"Duplicate sample id: {doc_id}")
            seen.add(doc_id)
            parse_tagged_input(row)
            if not isinstance(row["output"], str) or row["output"].strip().lower() not in STANCE_LABELS:
                raise ValueError("output must be favor, against or none.")
        except ValueError as error:
            row_id = row.get("id") if isinstance(row, dict) else None
            raise ValueError(f"Data validation failed: row_index={index}, id={row_id!r}: {error}") from error
    print(f"Validated {len(rows)} tagged Alpaca samples: {path}")
    return rows


def locate_step_and_answer_positions(
    generated_ids: torch.Tensor,
    tokenizer,
    expected_steps: int = 3,
) -> Tuple[Dict[str, int], int, str]:
    """
    定位教师生成结果中的：

    1. 每个 <STEP_END> 完成时的 token 位置；
    2. Stance 标签对应的 token 位置；
    3. 预测标签字符串。

    step_positions 中的位置表示：
        第 i 个 <STEP_END> 最后一个 token 的位置。

    这样提取的 hidden state 表示模型已经完整读完该推理步骤
    以及对应的结束边界。
    """
    token_ids = generated_ids.detach().cpu().tolist()

    if not token_ids:
        raise ValueError("generated_ids 为空。")

    decode_kwargs = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }

    # ======================================================
    # 逐 token 解码前缀
    #
    # prefix_texts[i] 表示 generated_ids[:i+1] 的完整文本。
    # 不使用逐 token piece 拼接，避免 byte/BPE 解码不一致。
    # ======================================================
    prefix_texts = []

    for token_index in range(len(token_ids)):
        prefix_text = tokenizer.decode(
            token_ids[:token_index + 1],
            **decode_kwargs,
        )
        prefix_texts.append(prefix_text)

    generated_text = prefix_texts[-1]

    # ======================================================
    # 1. 检查字符串层面的 marker 数量
    # ======================================================
    marker_matches = list(
        re.finditer(
            re.escape(STEP_END_MARKER),
            generated_text,
        )
    )

    if len(marker_matches) != expected_steps:
        raise ValueError(
            f"期望找到 {expected_steps} 个 "
            f"{STEP_END_MARKER}，实际找到 "
            f"{len(marker_matches)} 个。生成结果："
            f"{generated_text}"
        )

    # ======================================================
    # 2. 字符结束位置映射到 token 位置
    # ======================================================
    def char_end_to_token_index(
        char_end_exclusive: int,
    ) -> int:
        """
        char_end_exclusive 是字符区间的右开端点。

        返回第一个使解码前缀已经覆盖该字符区间的 token。
        """
        for token_index, prefix_text in enumerate(
            prefix_texts
        ):
            if len(prefix_text) >= char_end_exclusive:
                return token_index

        raise ValueError(
            f"字符位置 {char_end_exclusive} "
            "无法映射到 generated_ids。"
        )

    step_positions = {}

    for step_index, match in enumerate(marker_matches):
        # match.end() 是该 <STEP_END> 的字符右边界
        marker_end_token_pos = char_end_to_token_index(
            match.end()
        )

        step_positions[
            f"step_{step_index + 1}"
        ] = marker_end_token_pos

    # ======================================================
    # 3. 通过完整 JSON 文本定位 Stance 标签
    # ======================================================
    stance_match = re.search(
        r'"Stance"\s*:\s*"(favor|against|none)"',
        generated_text,
        flags=re.IGNORECASE,
    )

    if stance_match is None:
        raise ValueError(
            "没有找到合法的 Stance 字段。"
            "期望格式为："
            '"Stance": "favor/against/none"。'
            f"生成结果：{generated_text}"
        )

    predicted_label = stance_match.group(1).lower()

    # 标签字符范围，例如 against 的起点和终点
    label_char_start = stance_match.start(1)
    label_char_end = stance_match.end(1)

    # 第一个覆盖标签首字符的 token
    answer_start_token_pos = None

    for token_index, prefix_text in enumerate(prefix_texts):
        if len(prefix_text) > label_char_start:
            answer_start_token_pos = token_index
            break

    # 覆盖标签末字符的 token
    answer_end_token_pos = char_end_to_token_index(
        label_char_end
    )

    if answer_start_token_pos is None:
        raise ValueError(
            "无法定位 Stance 标签的起始 token。"
        )

    # 你的训练设定要求标签是单 token
    if answer_start_token_pos != answer_end_token_pos:
        label_token_ids = token_ids[
            answer_start_token_pos:
            answer_end_token_pos + 1
        ]

        raise ValueError(
            f"标签 {predicted_label!r} 在当前生成上下文中"
            f"占用了多个 token：{label_token_ids}。"
            "当前代码按单 token 标签进行 logits KD，"
            "需要调整标签格式或改为多 token 蒸馏。"
        )

    answer_token_pos = answer_start_token_pos

    # ======================================================
    # 4. 额外验证：定位到的 token 应包含标签文本
    # ======================================================
    answer_token_text = tokenizer.decode(
        [token_ids[answer_token_pos]],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    normalized_answer_token = re.sub(
        r"[^a-z]",
        "",
        answer_token_text.lower(),
    )

    if normalized_answer_token != predicted_label:
        raise ValueError(
            "Stance token 定位结果异常："
            f"label={predicted_label!r}, "
            f"token_pos={answer_token_pos}, "
            f"token_text={answer_token_text!r}, "
            f"token_id={token_ids[answer_token_pos]}。"
        )

    return (
        step_positions,
        answer_token_pos,
        predicted_label,
    )





def get_backbone(model):
    """
    获取 Qwen/LLaMA 等 CausalLM 的 transformer 主体，
    避免第二阶段为所有位置计算完整词表 logits。
    """
    if hasattr(model, "model"):
        return model.model

    if hasattr(model, "get_decoder"):
        return model.get_decoder()

    raise AttributeError(
        "无法找到模型 backbone，请检查模型是否具有 model 或 get_decoder()。"
    )


def move_inputs_to_model(model, encoded_inputs):
    """
    对 device_map='auto' 的模型，输入通常应放到 embedding 所在设备。
    """
    input_device = model.get_input_embeddings().weight.device

    return {
        key: value.to(input_device)
        for key, value in encoded_inputs.items()
        if isinstance(value, torch.Tensor)
    }


@torch.inference_mode()
def extract_teacher_step_states_two_stage(
    model,
    tokenizer,
    prompt_str: str,
    top_k: int = 50,
    max_new_tokens: int = 512,
    expected_steps: int = 3,
) -> Dict:
    model.eval()

    encoded = tokenizer(
        prompt_str,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model(model, encoded)

    prompt_len = inputs["input_ids"].shape[1]

    # ======================================================
    # 第一阶段：生成固定 CoT
    # ======================================================
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=False,
        output_hidden_states=False,
    )

    generated_ids = generated.sequences[
        0, prompt_len:
    ]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    (
        step_positions,
        answer_token_pos,
        predicted_label,
    ) = locate_step_and_answer_positions(
        generated_ids=generated_ids,
        tokenizer=tokenizer,
        expected_steps=expected_steps,
    )

    # ======================================================
    # 第二阶段：固定完整序列重新 forward
    # ======================================================
    full_input_ids = torch.cat(
        [
            inputs["input_ids"],
            generated_ids.unsqueeze(0).to(
                inputs["input_ids"].device
            ),
        ],
        dim=1,
    )

    generated_mask = torch.ones(
        (1, generated_ids.numel()),
        dtype=inputs["attention_mask"].dtype,
        device=inputs["attention_mask"].device,
    )

    full_attention_mask = torch.cat(
        [
            inputs["attention_mask"],
            generated_mask,
        ],
        dim=1,
    )

    # 只运行 backbone，不生成全序列词表 logits
    backbone = get_backbone(model)

    backbone_outputs = backbone(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )

    # 已包含最终 norm 后的最后一层状态
    final_hidden = backbone_outputs.last_hidden_state

    # ======================================================
    # 提取不同 CoT step 的最后一层状态
    # ======================================================
    teacher_step_hidden = {}

    for step_name, generated_pos in (
        step_positions.items()
    ):
        full_pos = prompt_len + generated_pos

        teacher_step_hidden[step_name] = (
            final_hidden[0, full_pos, :]
            .float()
            .cpu()
            .to(torch.float16)
        )

    # ======================================================
    # 提取预测标签前的决策状态和 logits
    # ======================================================
    answer_full_pos = prompt_len + answer_token_pos
    answer_predictor_full_pos = answer_full_pos - 1

    if answer_predictor_full_pos < 0:
        raise ValueError(
            "answer_predictor_full_pos 小于 0。"
        )

    answer_predictor_hidden_gpu = final_hidden[
        0,
        answer_predictor_full_pos,
        :,
    ]

    lm_head = model.get_output_embeddings()

    if lm_head is None:
        raise RuntimeError("模型没有 output embedding head。")

    head_parameter = next(lm_head.parameters())

    answer_logits = lm_head(
        answer_predictor_hidden_gpu.to(
            device=head_parameter.device,
            dtype=head_parameter.dtype,
        ).unsqueeze(0)
    )[0].float()

    topk_v, topk_i = torch.topk(
        answer_logits,
        k=top_k,
        dim=-1,
    )

    result = {
        "generated_text": generated_text,
        "generated_only_ids": (
            generated_ids.detach().cpu()
        ),

        "predicted_label": predicted_label,

        "step_positions": {
            name: int(position)
            for name, position in step_positions.items()
        },

        "answer_token_pos": int(answer_token_pos),
        "answer_token_id": int(
            generated_ids[answer_token_pos].item()
        ),
        "answer_predictor_full_pos": int(
            answer_predictor_full_pos
        ),

        "teacher_step_hidden": teacher_step_hidden,

        "answer_predictor_hidden": (
            answer_predictor_hidden_gpu
            .float()
            .cpu()
            .to(torch.float16)
        ),

        "topk_v": topk_v.cpu().to(torch.float16),
        "topk_i": topk_i.cpu().to(torch.int64),

        # 仅一整个词表向量，保留用于诊断
        "logits": answer_logits.cpu(),
    }

    del (
        generated,
        backbone_outputs,
        final_hidden,
        answer_logits,
        topk_v,
        topk_i,
    )

    return result


@torch.inference_mode()
def extract_teacher_step_states_one_pass(
    model,
    tokenizer,
    prompt_str: str,
    top_k: int = 50,
    max_new_tokens: int = 512,
    expected_steps: int = 3,
) -> Dict:
    model.eval()

    encoded = tokenizer(
        prompt_str,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model(model, encoded)

    prompt_len = inputs["input_ids"].shape[1]

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
        "output_logits": True,
    }

    try:
        generated = model.generate(**generation_kwargs)
    except (TypeError, ValueError) as error:
        # 兼容不支持 output_logits 的 Transformers 版本
        if "output_logits" not in str(error):
            raise

        generation_kwargs.pop("output_logits")
        generated = model.generate(**generation_kwargs)

    generated_ids = generated.sequences[
        0, prompt_len:
    ]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    (
        step_positions,
        answer_token_pos,
        predicted_label,
    ) = locate_step_and_answer_positions(
        generated_ids=generated_ids,
        tokenizer=tokenizer,
        expected_steps=expected_steps,
    )

    # ======================================================
    # 不同 CoT step 的最后一层状态
    #
    # generated token s 自身的状态出现在 generation step s+1
    # ======================================================
    teacher_step_hidden = {}

    for step_name, step_token_pos in (
        step_positions.items()
    ):
        next_generation_step = step_token_pos + 1

        if next_generation_step >= len(
            generated.hidden_states
        ):
            raise ValueError(
                f"{step_name} 后没有下一生成步，"
                "无法获得该 token 自身的 hidden state。"
            )

        state = generated.hidden_states[
            next_generation_step
        ][-1][0, -1, :]

        teacher_step_hidden[step_name] = (
            state.float().cpu().to(torch.float16)
        )

    # ======================================================
    # 预测标签的最后一层状态
    #
    # hidden_states[g] 用于预测 generated token g
    # ======================================================
    answer_predictor_hidden = (
        generated.hidden_states[
            answer_token_pos
        ][-1][0, -1, :]
    )

    if (
        hasattr(generated, "logits")
        and generated.logits is not None
    ):
        answer_logits = generated.logits[
            answer_token_pos
        ][0].float()
    else:
        answer_logits = generated.scores[
            answer_token_pos
        ][0].float()

    topk_v, topk_i = torch.topk(
        answer_logits,
        k=top_k,
        dim=-1,
    )

    result = {
        "generated_text": generated_text,
        "generated_only_ids": (
            generated_ids.detach().cpu()
        ),

        "predicted_label": predicted_label,

        "step_positions": {
            name: int(position)
            for name, position in step_positions.items()
        },

        "answer_token_pos": int(answer_token_pos),
        "answer_token_id": int(
            generated_ids[answer_token_pos].item()
        ),

        "teacher_step_hidden": teacher_step_hidden,

        "answer_predictor_hidden": (
            answer_predictor_hidden
            .float()
            .cpu()
            .to(torch.float16)
        ),

        "topk_v": topk_v.cpu().to(torch.float16),
        "topk_i": topk_i.cpu().to(torch.int64),
        "logits": answer_logits.cpu(),
    }

    del (
        generated,
        answer_logits,
        topk_v,
        topk_i,
    )

    return result


def tensor_difference_metrics(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> Dict[str, float]:
    a = tensor_a.detach().float().cpu().reshape(-1)
    b = tensor_b.detach().float().cpu().reshape(-1)

    diff = a - b

    cosine = F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=-1,
    ).item()

    relative_l2 = (
        diff.norm() / (b.norm() + 1e-12)
    ).item()

    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def logits_topk_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    k: int = 50,
) -> Dict[str, float]:
    a = logits_a.detach().float().cpu()
    b = logits_b.detach().float().cpu()

    top_a = torch.topk(a, k=k).indices
    top_b = torch.topk(b, k=k).indices

    overlap = torch.isin(top_a, top_b).float().mean().item()

    return {
        "top1_same": float(
            torch.argmax(a).item() == torch.argmax(b).item()
        ),
        f"top{k}_overlap": overlap,
    }




def numpy_or_tensor_to_cpu_tensor(
    value,
    dtype=None,
):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif not isinstance(value, torch.Tensor):
        value = torch.tensor(value)

    value = value.detach().cpu()

    if dtype is not None:
        value = value.to(dtype)

    return value


def save_teacher_sample(
    result: Dict,
    doc_id: str,
    save_dir: str,
    extract_mode: str,
    save_full_logits: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)

    teacher_step_hidden = {
        step_name: numpy_or_tensor_to_cpu_tensor(
            hidden,
            dtype=torch.float16,
        )
        for step_name, hidden in result[
            "teacher_step_hidden"
        ].items()
    }

    sample_data = {
        "doc_id": str(doc_id),
        "extract_mode": extract_mode,

        "generated_text": result["generated_text"],

        "generated_only_ids": (
            numpy_or_tensor_to_cpu_tensor(
                result["generated_only_ids"],
                dtype=torch.int32,
            )
        ),

        "predicted_label": result["predicted_label"],

        "step_positions": {
            name: int(position)
            for name, position in result[
                "step_positions"
            ].items()
        },

        "answer_token_pos": int(
            result["answer_token_pos"]
        ),
        "answer_token_id": int(
            result["answer_token_id"]
        ),

        # 教师不同 CoT step，统一最后一层
        "teacher_step_hidden": teacher_step_hidden,

        # 教师准备预测 stance 标签时的状态
        "answer_predictor_hidden": (
            numpy_or_tensor_to_cpu_tensor(
                result["answer_predictor_hidden"],
                dtype=torch.float16,
            )
        ),

        # 与原来的 logits KD 兼容
        "v": numpy_or_tensor_to_cpu_tensor(
            result["topk_v"],
            dtype=torch.float16,
        ),
        "i": numpy_or_tensor_to_cpu_tensor(
            result["topk_i"],
            dtype=torch.int64,
        ),
    }

    if (
        save_full_logits
        and result.get("logits") is not None
    ):
        sample_data["full_logits"] = (
            numpy_or_tensor_to_cpu_tensor(
                result["logits"],
                dtype=torch.float16,
            )
        )

    target_path = os.path.join(
        save_dir,
        f"{doc_id}.pt",
    )

    fd, temp_path = tempfile.mkstemp(
        suffix=".pt",
        dir=save_dir,
    )
    os.close(fd)

    try:
        torch.save(sample_data, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return target_path


# ==========================================================
# 配置
# ==========================================================
MODEL_PATH = "path/to/Qwen3-32B"
DATA_PATH = (
    "path/to/datasets/MT-CSD/sft_addlabel_noInstance_CUSTOM-LOSS_instruct/Biden/full_content_train.json"
)

OUTPUT_DIR = (
    "path/to/datasets/MT-CSD/teacher_extract_tagged/Biden"
)

SAMPLE_SAVE_DIR = os.path.join(
    OUTPUT_DIR,
    "teacher_samples",
)

TEXT_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "generated_texts.json",
)

FAILED_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "failed_samples.json",
)

TOP_K = 50
MAX_NEW_TOKENS = 512
SAVE_INTERVAL = 20

EXPECTED_STEPS = 3


# 可选："one_pass" 或 "two_stage"
EXTRACT_MODE = "two_stage"

# 完整词表 logits 很占空间，一般没有必要保存
SAVE_FULL_LOGITS = False


def atomic_save_json(data, path):
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temp_path, path)


def get_doc_id(sample, index):
    """Keep the exact dataset id for matching student kd_id and <id>.pt."""
    value = sample.get("id")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("id must be an integer or a nonempty string.")
    doc_id = str(value)
    if not doc_id or doc_id != doc_id.strip():
        raise ValueError("id must not be empty or have surrounding whitespace.")
    if doc_id in {".", ".."} or any(c in doc_id for c in '/\\:*?"<>|') or any(ord(c) < 32 for c in doc_id):
        raise ValueError("id contains characters unsafe for a cache filename.")
    return doc_id


def build_prompt_from_sample(sample, tokenizer):
    history_text, current_comment, target = parse_tagged_input(sample)

    # 组装 Prompt
    teacher_instruction = """
You are an expert in conversational stance detection.

Given a Target, a Current Comment, and its Conversation History, determine the stance expressed by the user of the Current Comment toward the Target.

Use exactly three concise reasoning steps.

Step 1 — Target and Reference Resolution:
Identify the Target and determine how the Current Comment refers to it. The reference may be direct or indirect, including an associated person, organization, product, action, policy, or event when the conversational context clearly connects it to the Target. Do not require the Target's exact name to appear.

Step 2 — Conversational Meaning:
Determine what the Current Comment is doing in context. Identify whether it supports, defends, praises, agrees with, criticizes, rejects, blames, mocks, questions, or sarcastically reverses a relevant claim. Use the Conversation History to resolve pronouns, reply relations, implicit meaning, and sarcasm. Focus on the stance of the Current Comment's user.

Step 3 — Directional Stance Decision:
Decide whether the overall evidence toward the Target is positive/supportive or negative/oppositional.

Use:

* "favor" when the comment expresses support, approval, defense, praise, agreement, preference, or a positive implication toward the Target;
* "against" when the comment expresses criticism, rejection, blame, ridicule, disapproval, opposition, or a negative implication toward the Target;
* "none" only when the comment contains neither supportive nor opposing evidence toward the Target, such as a purely unrelated, neutral, or genuinely indeterminate statement.

Important decision rules:

* Do not choose "none" merely because the stance is indirect, implicit, sarcastic, or expressed through an associated entity.
* When there is meaningful directional evidence, choose "favor" or "against".
* Use "none" only when both favor evidence and against evidence are absent.
* Sentiment is evidence, but the decision must be about the specified Target.
* Do not mention the final class label inside the Reasoning field.

Formatting requirements:

* Produce exactly one short statement for each step.
* End each step with the exact marker <STEP_END>.
* Keep the full reasoning under 100 words.
* Output exactly one valid JSON object with exactly two fields.
* Do not output Markdown fences or any text before or after the JSON object.

Required output structure:

{
"Reasoning": "<STEP_1>...<STEP_END><STEP_2>...<STEP_END><STEP_3>...<STEP_END>",
"Stance": "favor"
}

The Stance value must be exactly one of:
"favor", "against", or "none".
"""


    # user_prompt = f"Target: {target}\n\nConversation History: {h_text}\n\nCurrent Comment: {u_text}\n\nAnalyze the interaction and provide the structured output using the requested tags"
    student_style_input = (
        "[Conversation History]\n"
        f"{history_text}\n\n"
        "[Current Comment]\n"
        f"{current_comment}\n\n"
        "[Target]\n"
        f"{target}\n\n"
        "[Label]\n"
    )
    user_content = teacher_instruction + "\n\n" + student_style_input

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]


    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    
    return prompt_str


def load_existing_texts(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_existing_failures(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    dataset = load_and_validate_dataset(DATA_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)

    print(f"正在加载 Tokenizer：{MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"正在加载模型：{MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()

    generated_texts = load_existing_texts(TEXT_SAVE_PATH)
    failed_samples = load_existing_failures(FAILED_SAVE_PATH)

    success_count = 0
    skip_count = 0
    fail_count = 0

    print(f"提取模式：{EXTRACT_MODE}")
    print(f"样本数量：{len(dataset)}")
    print(f"保存目录：{SAMPLE_SAVE_DIR}")

    for index, sample in enumerate(
        tqdm(dataset, desc=f"Extracting {EXTRACT_MODE}")
    ):
        doc_id = get_doc_id(sample, index)

        sample_save_path = os.path.join(
            SAMPLE_SAVE_DIR,
            f"{doc_id}.pt",
        )

        # ======================================================
        # 断点续传
        # ======================================================
        if os.path.exists(sample_save_path):
            skip_count += 1
            continue

        try:
            prompt_str = build_prompt_from_sample(
                sample,
                tokenizer,
            )
            # print(prompt_str)

            # ==================================================
            # 调用 one-pass
            # ==================================================
            if EXTRACT_MODE == "one_pass":
                result = extract_teacher_step_states_one_pass(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_str=prompt_str,
                    top_k=TOP_K,
                    max_new_tokens=MAX_NEW_TOKENS,
                    expected_steps=EXPECTED_STEPS,
                )

            elif EXTRACT_MODE == "two_stage":
                result = extract_teacher_step_states_two_stage(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_str=prompt_str,
                    top_k=TOP_K,
                    max_new_tokens=MAX_NEW_TOKENS,
                    expected_steps=EXPECTED_STEPS,
                )

            else:
                raise ValueError(
                    f"未知 EXTRACT_MODE：{EXTRACT_MODE}"
                )


            # ==================================================
            # 保存单条 .pt
            # ==================================================
            save_teacher_sample(
                result=result,
                doc_id=doc_id,
                save_dir=SAMPLE_SAVE_DIR,
                extract_mode=EXTRACT_MODE,
                save_full_logits=SAVE_FULL_LOGITS,
            )

            # 单独保存文本，便于人工检查
            generated_texts[doc_id] = {
                "generated_text": result["generated_text"],
                "predicted_label": result["predicted_label"],
                "step_positions": result["step_positions"],
                "answer_token_pos": int(
                    result["answer_token_pos"]
                ),
                "extract_mode": EXTRACT_MODE,
            }

            # 若此前失败过，成功后移除失败记录
            failed_samples.pop(doc_id, None)

            success_count += 1

        except Exception as error:
            fail_count += 1

            failed_samples[doc_id] = {
                "index": index,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }

            print(
                f"\n[失败] doc_id={doc_id}, "
                f"error={repr(error)}"
            )

        finally:
            gc.collect()
            torch.cuda.empty_cache()

        # ======================================================
        # 周期性保存索引文件
        # ======================================================
        handled_count = success_count + fail_count

        if (
            handled_count > 0
            and handled_count % SAVE_INTERVAL == 0
        ):
            atomic_save_json(
                generated_texts,
                TEXT_SAVE_PATH,
            )

            atomic_save_json(
                failed_samples,
                FAILED_SAVE_PATH,
            )

    # 最终保存
    atomic_save_json(
        generated_texts,
        TEXT_SAVE_PATH,
    )

    atomic_save_json(
        failed_samples,
        FAILED_SAVE_PATH,
    )

    print("\n提取完成")
    print(f"新成功样本：{success_count}")
    print(f"跳过已有样本：{skip_count}")
    print(f"失败样本：{fail_count}")
    print(f"结果目录：{SAMPLE_SAVE_DIR}")


if __name__ == "__main__":
    main()