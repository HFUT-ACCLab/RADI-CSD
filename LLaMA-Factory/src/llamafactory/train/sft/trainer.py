# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import numpy as np
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional,Union, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer, TrainerCallback
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ..trainer_utils import report_score

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments, DataArguments

from collections import OrderedDict
logger = logging.get_logger(__name__)

def get_trainable_projector_parameter(
    projector: nn.Module,
):
    """
    获取投影头中实际参与训练的参考参数。

    对 PEFT ModulesToSaveWrapper：
    - original_module 通常被冻结；
    - modules_to_save.default 才是活动、可训练副本。
    """
    for parameter_name, parameter in (
        projector.named_parameters()
    ):
        if parameter.requires_grad:
            return parameter_name, parameter

    raise RuntimeError(
        f"投影头 {projector.__class__.__name__} "
        "中没有可训练参数。"
    )


def apply_kd_projector(
    projector: nn.Module,
    student_vector: torch.Tensor,
) -> torch.Tensor:
    """
    将学生 hidden 转为活动投影头的实际 dtype/device，
    然后执行投影。
    """
    parameter_name, reference_parameter = (
        get_trainable_projector_parameter(
            projector
        )
    )

    student_vector = student_vector.to(
        device=reference_parameter.device,
        dtype=reference_parameter.dtype,
    )

    projected = projector(student_vector)

    # 后续 cosine loss 使用 FP32 更稳定
    return projected.float()


def compute_answer_topk_kd(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    predictor_positions: torch.Tensor,
    first_label_positions: torch.Tensor,
    teacher_batch,
    label_id_to_name,
    temperature: float = 2.0,
):
    """
    student_logits:
        [batch, seq_len, student_vocab]

    教师 v/i:
        [top_k]
    """
    losses = []

    batch_size = student_logits.shape[0]
    vocab_size = student_logits.shape[-1]

    for batch_index in range(batch_size):
        teacher_data = teacher_batch[batch_index]

        if teacher_data is None:
            continue

        gold_token_id = int(
            labels[
                batch_index,
                first_label_positions[batch_index],
            ].item()
        )

        gold_label_name = label_id_to_name.get(
            gold_token_id
        )

        if gold_label_name is None:
            continue

        teacher_label = str(
            teacher_data.get(
                "predicted_label",
                "",
            )
        ).strip().lower()

        # 教师做错时不蒸馏
        if teacher_label != gold_label_name:
            continue

        if (
            "v" not in teacher_data
            or "i" not in teacher_data
        ):
            continue

        teacher_values = torch.as_tensor(
            teacher_data["v"],
            dtype=torch.float32,
        ).reshape(-1)

        teacher_indices = torch.as_tensor(
            teacher_data["i"],
            dtype=torch.long,
        ).reshape(-1)

        if teacher_values.numel() == 0:
            continue

        if teacher_values.shape != teacher_indices.shape:
            continue

        # 教师和学生必须共享词表 ID
        if (
            teacher_indices.min().item() < 0
            or teacher_indices.max().item()
            >= vocab_size
        ):
            continue

        teacher_answer_token_id = teacher_data.get(
            "answer_token_id"
        )

        if teacher_answer_token_id is not None:
            teacher_answer_token_id = int(
                teacher_answer_token_id
            )

            # 用学生 tokenizer 解码教师 token id，
            # 简单检查词表兼容性
            mapped_teacher_name = (
                label_id_to_name.get(
                    teacher_answer_token_id
                )
            )

            if (
                mapped_teacher_name is not None
                and mapped_teacher_name
                != teacher_label
            ):
                continue

        student_vector = student_logits[
            batch_index,
            predictor_positions[batch_index],
            :,
        ]

        teacher_indices = teacher_indices.to(
            student_vector.device
        )

        teacher_values = teacher_values.to(
            student_vector.device
        )

        teacher_log_probs = F.log_softmax(
            teacher_values / temperature,
            dim=-1,
        )

        teacher_probs = teacher_log_probs.exp()
####################
        student_full_log_probs = F.log_softmax(
            student_vector.float() / temperature,
            dim=-1,
        )

        student_selected_log_probs = (
            student_full_log_probs.gather(
                dim=0,
                index=teacher_indices,
            )
        )
        # student_selected_logits = student_vector.float().gather(
        #     dim=0,
        #     index=teacher_indices,
        # )

        # student_selected_log_probs = F.log_softmax(
        #     student_selected_logits / temperature,
        #     dim=-1,
        # )
##############################

        sample_loss = torch.sum(
            teacher_probs
            * (
                teacher_log_probs
                - student_selected_log_probs
            )
        ) * (temperature ** 2)


        losses.append(sample_loss)

    if not losses:
        return student_logits.sum() * 0.0

    return torch.stack(losses).mean()


def normalized_cosine_loss(
    student_vector: torch.Tensor,
    teacher_vector: torch.Tensor,
):
    student_vector = F.normalize(
        student_vector.float(),
        p=2,
        dim=-1,
    )

    teacher_vector = F.normalize(
        teacher_vector.float(),
        p=2,
        dim=-1,
    )

    return (
        1.0
        - F.cosine_similarity(
            student_vector,
            teacher_vector,
            dim=-1,
        )
    ).mean()


def compute_step_hidden_kd(
    student_step_features,
    teacher_batch,
    projectors,
    gold_label_names,
):
    step_losses = {}
    valid_step_losses = []

    for step_name in [
        "step_1",
        "step_2",
        "step_3",
    ]:
        if step_name not in student_step_features:
            continue

        sample_losses = []

        student_batch_vectors = (
            student_step_features[step_name]
        )

        projector = projectors[step_name]


        for batch_index, teacher_data in enumerate(
            teacher_batch
        ):
            if teacher_data is None:
                continue

            gold_label = gold_label_names[
                batch_index
            ]

            if gold_label is None:
                continue

            teacher_label = str(
                teacher_data.get(
                    "predicted_label",
                    "",
                )
            ).strip().lower()

            # 教师预测错误时，不使用其 CoT 特征
            if teacher_label != gold_label:
                continue

            teacher_steps = teacher_data.get(
                "teacher_step_hidden",
                {},
            )

            teacher_vector = teacher_steps.get(
                step_name
            )

            if teacher_vector is None:
                continue

            student_vector = (
                student_batch_vectors[
                    batch_index:
                    batch_index + 1
                ]
            )

            student_projected = apply_kd_projector(
                projector=projector,
                student_vector=student_vector,
            )


            teacher_vector = torch.as_tensor(
                teacher_vector,
                dtype=torch.float32,
                device=student_projected.device,
            ).reshape(1, -1)

            if (
                student_projected.shape[-1]
                != teacher_vector.shape[-1]
            ):
                continue

            sample_loss = normalized_cosine_loss(
                student_projected,
                teacher_vector,
            )

            sample_losses.append(sample_loss)

        if sample_losses:
            step_loss = torch.stack(
                sample_losses
            ).mean()

            step_losses[step_name] = step_loss
            valid_step_losses.append(step_loss)

    if not valid_step_losses:
        reference = next(
            iter(student_step_features.values())
        )

        zero = reference.sum() * 0.0
        return zero, step_losses

    total_step_loss = torch.stack(
        valid_step_losses
    ).mean()

    return total_step_loss, step_losses
def compute_answer_hidden_kd(
    student_answer_hidden,
    teacher_batch,
    answer_projector,
    gold_label_names,
):
    sample_losses = []


    for batch_index, teacher_data in enumerate(
        teacher_batch
    ):
        if teacher_data is None:
            continue

        gold_label = gold_label_names[
            batch_index
        ]

        if gold_label is None:
            continue

        teacher_label = str(
            teacher_data.get(
                "predicted_label",
                "",
            )
        ).strip().lower()

        if teacher_label != gold_label:
            continue

        teacher_vector = teacher_data.get(
            "answer_predictor_hidden"
        )

        if teacher_vector is None:
            continue

        student_vector = student_answer_hidden[
            batch_index:
            batch_index + 1
        ]

        student_projected = apply_kd_projector(
            projector=answer_projector,
            student_vector=student_vector,
        )

        teacher_vector = torch.as_tensor(
            teacher_vector,
            dtype=torch.float32,
            device=student_projected.device,
        ).reshape(1, -1)

        if (
            student_projected.shape[-1]
            != teacher_vector.shape[-1]
        ):
            continue

        sample_loss = normalized_cosine_loss(
            student_projected,
            teacher_vector,
        )

        sample_losses.append(sample_loss)

    if not sample_losses:
        return student_answer_hidden.sum() * 0.0

    return torch.stack(sample_losses).mean()

def unwrap_causal_lm(model):
    current = model

    while hasattr(current, "module"):
        current = current.module

    if hasattr(current, "get_base_model"):
        current = current.get_base_model()

    return current


def get_student_backbone(model):
    causal_lm = unwrap_causal_lm(model)
    current = causal_lm

    visited = set()

    for _ in range(10):
        if id(current) in visited:
            break

        visited.add(id(current))

        if (
            hasattr(current, "layers")
            and isinstance(
                current.layers,
                torch.nn.ModuleList,
            )
        ):
            backbone = current
            layers = current.layers
            final_norm = getattr(
                current,
                "norm",
                None,
            )

            return (
                causal_lm,
                backbone,
                layers,
                final_norm,
            )

        if hasattr(current, "model"):
            current = current.model
            continue

        if hasattr(current, "base_model"):
            current = current.base_model
            continue

        break

    raise RuntimeError(
        "无法找到学生模型的 Transformer layers。"
    )
def infer_teacher_hidden_size(
    kd_dir: str,
) -> int:
    for filename in sorted(os.listdir(kd_dir)):
        if not filename.endswith(".pt"):
            continue

        path = os.path.join(kd_dir, filename)

        try:
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            continue

        hidden = data.get(
            "answer_predictor_hidden"
        )

        if hidden is not None:
            return int(
                torch.as_tensor(hidden).numel()
            )

    raise RuntimeError(
        f"无法从 {kd_dir} 推断教师 hidden size。"
    )



class TeacherPTCache:
    def __init__(
        self,
        kd_dir: str,
        max_items: int = 512,
    ):
        self.kd_dir = kd_dir
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(
        self,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        doc_id = str(doc_id)

        if doc_id in self.cache:
            data = self.cache.pop(doc_id)
            self.cache[doc_id] = data
            return data

        path = os.path.join(
            self.kd_dir,
            f"{doc_id}.pt",
        )

        if not os.path.exists(path):
            return None

        try:
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            data = torch.load(
                path,
                map_location="cpu",
            )

        self.cache[doc_id] = data

        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return data
class SeparateLossPlotCallback(TrainerCallback):
    """
    训练结束后，自动从 state.log_history 中提取自定义 loss，
    并分别保存单项 loss 图、KD 组件汇总图、hidden step loss 汇总图。
    """

    def on_train_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        def smooth_curve(points, factor=0.85):
            smoothed = []
            for point in points:
                if smoothed:
                    smoothed.append(smoothed[-1] * factor + point * (1 - factor))
                else:
                    smoothed.append(point)
            return smoothed

        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return None

        # ======================================================
        # 1. 从 log_history 中自动收集所有曲线
        # ======================================================
        metric_history = {}

        wanted_prefixes = (
            "loss_sft",
            "loss_step_kd",
            "loss_answer_hidden_kd",
            "loss_logit_kd",
            "loss_hidden_",
            "kd_scale",
        )

        # HF 默认 total loss 字段叫 "loss"
        wanted_exact = {
            "loss",
            "eval_loss",
            "loss_total_manual",
            "loss_weighted_kd",
            "loss_sft",
            "loss_step_kd",
            "loss_answer_hidden_kd",
            "loss_logit_kd",
            "loss_step_kd_weighted",
            "loss_answer_hidden_kd_weighted",
            "loss_logit_kd_weighted",
            "kd_scale",
        }


        for log in state.log_history:
            if "step" not in log:
                continue

            step = log["step"]

            for key, value in log.items():
                if key == "step":
                    continue

                is_needed = (
                    key in wanted_exact
                    or key.startswith("loss_hidden_")
                )

                if not is_needed:
                    continue

                value = safe_float(value)
                if value is None:
                    continue

                metric_history.setdefault(key, []).append(
                    {
                        "step": int(step),
                        "value": value,
                    }
                )

        if not metric_history:
            print("⚠️ 没有在 state.log_history 中找到自定义 loss，跳过画图。")
            return

        # ======================================================
        # 2. 保存原始 loss 数据，方便后续论文分析
        # ======================================================
        history_path = os.path.join(output_dir, "custom_loss_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(metric_history, f, indent=2, ensure_ascii=False)

        # ======================================================
        # 3. 单项曲线绘图
        # ======================================================
        def plot_single_metric(metric_name, records):
            if not records:
                return

            steps = [item["step"] for item in records]
            values = [item["value"] for item in records]

            if len(steps) < 2:
                return

            plt.figure(figsize=(8, 5))
            plt.plot(steps, values, alpha=0.25, label="Original")
            plt.plot(steps, smooth_curve(values), linewidth=2, label="Smoothed")
            plt.xlabel("Steps", fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.title(metric_name, fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            filename = f"training_curve_{metric_name}.png"
            filename = filename.replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

        for metric_name, records in metric_history.items():
            plot_single_metric(metric_name, records)

        # ======================================================
        # 4. 多项 loss 汇总图
        # ======================================================
        def plot_multi_metrics(metric_names, title, filename):
            valid_metrics = [
                name for name in metric_names
                if name in metric_history and len(metric_history[name]) >= 2
            ]

            if not valid_metrics:
                return

            plt.figure(figsize=(10, 6))

            for name in valid_metrics:
                records = metric_history[name]
                steps = [item["step"] for item in records]
                values = [item["value"] for item in records]

                plt.plot(
                    steps,
                    smooth_curve(values),
                    linewidth=2,
                    label=name,
                )

            plt.xlabel("Steps", fontsize=12)
            plt.ylabel("Loss / Value", fontsize=12)
            plt.title(title, fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

        # KD 总组件汇总图
        plot_multi_metrics(
            [
                "loss_sft",
                "loss_step_kd",
                "loss_answer_hidden_kd",
                "loss_logit_kd",
            ],
            "SFT and KD Loss Components",
            "training_curve_kd_components.png",
        )

        # hidden step 子 loss 汇总图
        hidden_metric_names = sorted(
            [
                name for name in metric_history.keys()
                if name.startswith("loss_hidden_")
            ]
        )

        plot_multi_metrics(
            hidden_metric_names,
            "Hidden Step KD Losses",
            "training_curve_hidden_steps.png",
        )

        print(f"\n✅ 已保存自定义 loss 曲线和原始 loss 数据到: {output_dir}")



class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.add_callback(SeparateLossPlotCallback())
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func
        processing_object = getattr(
            self,
            "processing_class",
            None,
        )

        if processing_object is None:
            processing_object = getattr(
                self,
                "tokenizer",
                None,
            )

        # 若传入的是 ProcessorMixin，则取它内部的 tokenizer
        tokenizer = getattr(
            processing_object,
            "tokenizer",
            processing_object,
        )

        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise RuntimeError(
                "无法从 Trainer 中获得文本 tokenizer。"
            )

        self.kd_tokenizer = tokenizer
        self.label_id_to_name = self._build_label_map(
            tokenizer
        )
        self.custom_kd_accumulator = {}
        self.custom_kd_steps = 0
####################验证student hidden中的参数是否更新#################################
        # self.kd_debug_grad = False
        # self.kd_debug_every = 1
        self.kd_debug_grad = bool(
            getattr(finetuning_args, "kd_debug_grad", False)
        )

        self.kd_debug_every = int(
            getattr(finetuning_args, "kd_debug_every", 50)
        )

        self._last_kd_tensor_grad_norms = {}

        self._initialize_kd()


    def _record_kd_tensor_grad(
        self,
        name: str,
        grad: torch.Tensor,
    ):
        grad_norm = (
            grad.detach()
            .float()
            .norm()
            .item()
        )

        self._last_kd_tensor_grad_norms[
            f"debug_grad_{name}"
        ] = grad_norm

        if (
            self.args.local_rank <= 0
            and self.state.global_step
            % self.kd_debug_every == 0
        ):
            print(
                f"[KD Grad Debug][Step {self.state.global_step}] "
                f"{name} grad_norm={grad_norm:.6e}"
            )
####################################################
    def _validate_projectors(
        self,
        teacher_hidden_size: int,
    ):
        student_hidden_size = int(
            self.causal_lm.config.hidden_size
        )

        for name, projector in self.kd_projectors.items():
            parameters = list(projector.parameters())

            if not parameters:
                raise RuntimeError(
                    f"KD 投影头 {name} 没有任何参数。"
                )

            if not any(
                parameter.requires_grad
                for parameter in parameters
            ):
                raise RuntimeError(
                    f"KD 投影头 {name} 全部被冻结，"
                    "请检查 additional_target。"
                )

            linear_layers = [
                module
                for module in projector.modules()
                if isinstance(module, nn.Linear)
            ]

            if not linear_layers:
                raise RuntimeError(
                    f"KD 投影头 {name} 中没有 nn.Linear。"
                )

            # PEFT ModulesToSaveWrapper 可能包含多个副本，
            # 只要所有 Linear 的维度正确即可。
            for linear in linear_layers:
                if linear.in_features != student_hidden_size:
                    raise RuntimeError(
                        f"{name} 输入维度错误："
                        f"{linear.in_features}，"
                        f"预期 {student_hidden_size}"
                    )

                if linear.out_features != teacher_hidden_size:
                    raise RuntimeError(
                        f"{name} 输出维度错误："
                        f"{linear.out_features}，"
                        f"预期 {teacher_hidden_size}"
                    )

            logger.info_rank0(
                f"Validated projector {name}: "
                f"{student_hidden_size} -> "
                f"{teacher_hidden_size}, "
                f"trainable_params="
                f"{sum(p.numel() for p in parameters if p.requires_grad)}"
            )

        
    def _build_label_map(self, tokenizer):
        label_id_to_name = {}
        label_name_to_ids = {
            "favor": [],
            "against": [],
            "none": [],
        }

        for label_name in label_name_to_ids:
            for text in [
                label_name,
                " " + label_name,
            ]:
                token_ids = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )

                if len(token_ids) == 1:
                    token_id = int(token_ids[0])

                    label_id_to_name[token_id] = (
                        label_name
                    )

                    label_name_to_ids[
                        label_name
                    ].append(token_id)

        missing_labels = [
            label_name
            for label_name, token_ids
            in label_name_to_ids.items()
            if not token_ids
        ]

        if missing_labels:
            raise RuntimeError(
                "以下标签不是单 token："
                f"{missing_labels}。"
                "当前单 token KD 逻辑不能使用。"
            )

        logger.info_rank0(
            f"Student label token map: "
            f"{label_id_to_name}"
        )

        return label_id_to_name

    def _initialize_kd(self):
        self.kd_dir = getattr(
            self.finetuning_args,
            "kd_dir",
            None,
        )

        self.kd_enabled = bool(self.kd_dir)

        if not self.kd_enabled:
            logger.info_rank0(
                "kd_dir 未配置，使用纯 SFT。"
            )
            return

        if not os.path.isdir(self.kd_dir):
            raise ValueError(
                f"kd_dir 不存在：{self.kd_dir}"
            )

        self.teacher_cache = TeacherPTCache(
            kd_dir=self.kd_dir,
            max_items=512,
        )

        (
            causal_lm,
            student_backbone,
            student_layers,
            student_final_norm,
        ) = get_student_backbone(self.model)

        self.causal_lm = causal_lm
        self.student_backbone = student_backbone
        self.student_layers = student_layers
        self.student_final_norm = student_final_norm

        if self.student_final_norm is None:
            raise RuntimeError(
                "无法找到学生最终 norm。"
            )

        num_layers = len(student_layers)

        self.step_layer_map = {
            "step_1": max(
                0,
                round(num_layers * 0.25) - 1,
            ),
            "step_2": max(
                0,
                round(num_layers * 0.5) - 1,
            ),
            "step_3": max(
                0,
                round(num_layers * 0.75) - 1,
            ),
        }

        projector_names = {
            "step_1": "proj_step_1",
            "step_2": "proj_step_2",
            "step_3": "proj_step_3",
            "answer": "proj_answer",
        }

        self.kd_projectors = {}

        for kd_name, module_name in (
            projector_names.items()
        ):
            if not hasattr(causal_lm, module_name):
                raise RuntimeError(
                    f"模型缺少投影头 {module_name}。"
                )

            self.kd_projectors[kd_name] = getattr(
                causal_lm,
                module_name,
            )

        teacher_dim_from_pt = (
            infer_teacher_hidden_size(self.kd_dir)
        )

        logger.info_rank0(
            f"KD student layers: {self.step_layer_map}"
        )
        logger.info_rank0(
            f"Teacher hidden size from PT: "
            f"{teacher_dim_from_pt}"
        )

        self._validate_projectors(
            teacher_dim_from_pt
        )

    @override
    def log(
        self,
        logs: dict[str, float],
        *args,
        **kwargs,
    ):
        if (
            hasattr(
                self,
                "custom_kd_accumulator",
            )
            and self.custom_kd_steps > 0
        ):
            for key, total_value in (
                self.custom_kd_accumulator.items()
            ):
                logs[key] = (
                    total_value
                    / self.custom_kd_steps
                )

            self.custom_kd_accumulator = {}
            self.custom_kd_steps = 0

        if hasattr(self, "_last_kd_tensor_grad_norms"):
            for key, value in (
                self._last_kd_tensor_grad_norms.items()
            ):
                logs[key] = value

            self._last_kd_tensor_grad_norms = {}

        super().log(logs, *args, **kwargs)
       

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)





    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        # ======================================================
        # 1. 取出 KD ID 和备份 labels
        # ======================================================
        kd_ids = inputs.pop("kd_id", None)
        if (
            self.kd_enabled
            and model.training
            and kd_ids is None
        ):
            raise RuntimeError(
                "已配置 kd_dir，但训练 batch 中没有 kd_id。"
                "请检查数据预处理、data collator，"
                "以及 remove_unused_columns=False。"
            )
        labels = inputs.get("labels")

        if labels is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )

        labels_backup = labels.clone()
        
        batch_size = labels_backup.shape[0]
        device = labels_backup.device
        
        # ======================================================
        # 2. 定位学生标签 token 和标签预测位置
        # ======================================================
        label_mask = labels_backup.ne(-100)

        has_valid_label = label_mask.any(dim=1)

        first_label_positions = (
            label_mask.long().argmax(dim=1)
        )
        # #############打印输入内容
        # if (
        #     model.training
        #     and self.args.local_rank <= 0
        #     and self.state.global_step % 1 == 0
        # ):
        #     # text = self.kd_tokenizer.decode(
        #     #     inputs["input_ids"][0].detach().cpu().tolist(),
        #     #     skip_special_tokens=False,
        #     # )
        #     # print("\n[FULL INPUT]")
        #     # print(text)
        #     for i in range(min(2, batch_size)):
        #         pos = int(first_label_positions[i].item())
        #         token_id = int(labels_backup[i, pos].item())
        #         token_text = self.kd_tokenizer.decode(
        #             [token_id],
        #             skip_special_tokens=False,
        #         )
        #         print(
        #             f"[KD Label Debug] sample={i}, "
        #             f"first_label_pos={pos}, "
        #             f"token_id={token_id}, "
        #             f"token_text={repr(token_text)}, "
        #             f"mapped_label={self.label_id_to_name.get(token_id)}"
        #         )
        # #############

        predictor_positions = torch.clamp(
            first_label_positions - 1,
            min=0,
        )

        batch_indices = torch.arange(
            batch_size,
            device=device,
        )

        gold_token_ids = labels_backup[
            batch_indices,
            first_label_positions,
        ]

        gold_label_names = []

        for batch_index in range(batch_size):
            if not bool(
                has_valid_label[batch_index].item()
            ):
                gold_label_names.append(None)
                continue

            token_id = int(
                gold_token_ids[batch_index].item()
            )

            gold_label_names.append(
                self.label_id_to_name.get(token_id)
            )

        # ======================================================
        # 3. 加载教师 .pt
        # ======================================================
        teacher_batch = [None] * batch_size

        if kd_ids is not None:
            if isinstance(kd_ids, torch.Tensor):
                raw_ids = kd_ids.detach().cpu().tolist()
            else:
                raw_ids = list(kd_ids)

            for batch_index, doc_id in enumerate(raw_ids):
                teacher_batch[batch_index] = (
                    self.teacher_cache.get(
                        str(doc_id)
                    )
                )

        use_kd = (
            self.kd_enabled
            and model.training
            and kd_ids is not None
            and any(
                item is not None
                for item in teacher_batch
            )
        )


                # ======================================================
        # 4. 正常前向：不再使用 Hook，改为直接返回 hidden_states
        # ======================================================
        if use_kd:
            inputs["output_hidden_states"] = True
            inputs["use_cache"] = False

        sft_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            **kwargs,
        )

        # ======================================================
        # 5. 从 outputs.hidden_states 中提取学生 step hidden
        #    和 answer hidden
        # ======================================================
        student_step_features = {}
        student_answer_hidden = None

        if use_kd:
            hidden_states = getattr(
                outputs,
                "hidden_states",
                None,
            )

            if hidden_states is None and isinstance(outputs, dict):
                hidden_states = outputs.get("hidden_states", None)

            if hidden_states is None:
                raise RuntimeError(
                    "outputs.hidden_states is None。"
                    "请确认 inputs['output_hidden_states']=True 已生效。"
                )

            # hidden_states[0] 通常是 embedding 后 hidden
            # hidden_states[layer_index + 1] 通常对应第 layer_index 层输出
            # hidden_states[-1] 是最终 norm 后 hidden
            for step_name, layer_index in self.step_layer_map.items():
                hidden_state_index = layer_index + 1

                if hidden_state_index >= len(hidden_states):
                    raise IndexError(
                        f"{step_name} 对应学生层 {layer_index}，"
                        f"需要 hidden_states[{hidden_state_index}]，"
                        f"但 hidden_states 长度只有 {len(hidden_states)}。"
                    )

                layer_hidden = hidden_states[hidden_state_index]

                current_positions = predictor_positions.to(
                    layer_hidden.device
                )

                current_batch = torch.arange(
                    layer_hidden.shape[0],
                    device=layer_hidden.device,
                )

                feature = layer_hidden[
                    current_batch,
                    current_positions,
                    :,
                ]

                student_step_features[step_name] = feature

                if (
                    self.kd_debug_grad
                    and self.args.local_rank <= 0
                    and self.state.global_step % self.kd_debug_every == 0
                ):
                    print(
                        f"[KD Grad Debug][Step {self.state.global_step}] "
                        f"{step_name}: "
                        f"requires_grad={feature.requires_grad}, "
                        f"grad_fn={feature.grad_fn}"
                    )

                    if feature.requires_grad:
                        feature.register_hook(
                            lambda grad, name=step_name: (
                                self._record_kd_tensor_grad(
                                    f"hidden_{name}",
                                    grad,
                                )
                            )
                        )

            final_hidden = hidden_states[-1]

            current_positions = predictor_positions.to(
                final_hidden.device
            )

            current_batch = torch.arange(
                final_hidden.shape[0],
                device=final_hidden.device,
            )

            student_answer_hidden = final_hidden[
                current_batch,
                current_positions,
                :,
            ]

            if (
                self.kd_debug_grad
                and self.args.local_rank <= 0
                and self.state.global_step % self.kd_debug_every == 0
            ):
                print(
                    f"[KD Grad Debug][Step {self.state.global_step}] "
                    f"answer_hidden: "
                    f"requires_grad={student_answer_hidden.requires_grad}, "
                    f"grad_fn={student_answer_hidden.grad_fn}"
                )

                if student_answer_hidden.requires_grad:
                    student_answer_hidden.register_hook(
                        lambda grad: (
                            self._record_kd_tensor_grad(
                                "answer_hidden",
                                grad,
                            )
                        )
                    )


        def record_custom_losses(metric_values):
            if self.args.local_rank > 0:
                return

            if not hasattr(self, "custom_kd_accumulator"):
                self.custom_kd_accumulator = {}
                self.custom_kd_steps = 0

            for key, value in metric_values.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().item()
                else:
                    value = float(value)

                self.custom_kd_accumulator[key] = (
                    self.custom_kd_accumulator.get(key, 0.0) + value
                )

            self.custom_kd_steps += 1
        if not use_kd:
            zero_value = float(sft_loss.detach().float().item()) * 0.0

            record_custom_losses(
                {
                    "loss_sft": sft_loss,
                    "loss_step_kd": zero_value,
                    "loss_answer_hidden_kd": zero_value,
                    "loss_logit_kd": zero_value,
                    "kd_scale": zero_value,
                }
            )

            return (
                (sft_loss, outputs)
                if return_outputs
                else sft_loss
            )
        # ======================================================
        # 6. Step hidden KD
        # ======================================================
        step_kd_loss, per_step_losses = (
            compute_step_hidden_kd(
                student_step_features=(
                    student_step_features
                ),
                teacher_batch=teacher_batch,
                projectors=self.kd_projectors,
                gold_label_names=gold_label_names,
            )
        )

        # ======================================================
        # 7. Answer hidden KD
        # ======================================================
        if student_answer_hidden is None:
            answer_hidden_kd_loss = sft_loss * 0.0
        else:
            answer_hidden_kd_loss = compute_answer_hidden_kd(
                student_answer_hidden=student_answer_hidden,
                teacher_batch=teacher_batch,
                answer_projector=self.kd_projectors["answer"],
                gold_label_names=gold_label_names,
            )


        # ======================================================
        # 8. Answer Top-K logit KD
        # ======================================================
        logit_kd_loss = compute_answer_topk_kd(
            student_logits=outputs.logits,
            labels=labels_backup,
            predictor_positions=predictor_positions,
            first_label_positions=(
                first_label_positions
            ),
            teacher_batch=teacher_batch,
            label_id_to_name=(
                self.label_id_to_name
            ),
            temperature=2.0,
        )

        # ======================================================
        # 9. 加权总损失
        # ======================================================
        lambda_step = 0.5
        lambda_answer = 0.5
        lambda_logit = 0.5

        # 前期渐进打开 KD，避免训练刚开始时不稳定
        kd_warmup_steps = 200

        kd_scale = min(
            1.0,
            float(self.state.global_step + 1)
            / float(max(kd_warmup_steps, 1)),
        )

        total_loss = (
            sft_loss
            + kd_scale * (
                lambda_step * step_kd_loss
                + lambda_answer
                * answer_hidden_kd_loss
                + lambda_logit
                * logit_kd_loss
            )
        )
        ###without kd
        # total_loss = (
        #     sft_loss
        #     + kd_scale * (
        #         lambda_step * step_kd_loss
        #         + lambda_answer
        #         * answer_hidden_kd_loss
        #     )
        # )
        #####without feat
        # total_loss = (
        #     sft_loss
        #     + kd_scale * (
        #         + lambda_logit
        #         * logit_kd_loss
        #     )
        # )
        ##########without feat_step
        # total_loss = (
        #             sft_loss
        #             + kd_scale * (
        #                 # lambda_step * step_kd_loss
        #                 + lambda_answer * answer_hidden_kd_loss
        #                 + lambda_logit * logit_kd_loss
        #             )
        #         )
        #######without ce
        # total_loss = (
        #     kd_scale * (
        #         lambda_step * step_kd_loss
        #         + lambda_answer
        #         * answer_hidden_kd_loss
        #         + lambda_logit
        #         * logit_kd_loss
        #     )
        # )
        # ======================================================
        # 10. 记录日志
        # ======================================================
        weighted_kd_loss = kd_scale * (
             lambda_step * step_kd_loss
            + lambda_answer * answer_hidden_kd_loss
            + lambda_logit * logit_kd_loss
        )

        metric_values = {
            "loss_total_manual": total_loss,
            "loss_weighted_kd": weighted_kd_loss,
            "loss_sft": sft_loss,
            "loss_step_kd": step_kd_loss,
            "loss_answer_hidden_kd": answer_hidden_kd_loss,
            "loss_logit_kd": logit_kd_loss,
            "loss_step_kd_weighted": kd_scale * lambda_step * step_kd_loss,
            "loss_answer_hidden_kd_weighted": kd_scale * lambda_answer * answer_hidden_kd_loss,
            "loss_logit_kd_weighted": kd_scale * lambda_logit * logit_kd_loss,
            "kd_scale": kd_scale,
        }


        for step_name, step_loss in per_step_losses.items():
            metric_values[f"loss_hidden_{step_name}"] = step_loss

        record_custom_losses(metric_values)
        if (
            self.args.local_rank <= 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            print(
                f"[Step {self.state.global_step}] "
                f"SFT={sft_loss.item():.4f} | "
                f"StepKD={step_kd_loss.item():.4f} | "
                f"AnswerKD={answer_hidden_kd_loss.item():.4f} | "
                f"LogitKD={logit_kd_loss.item():.4f} | "
                f"Scale={kd_scale:.3f} | "
                f"Total={total_loss.item():.4f}"
            )


        return (
            (total_loss, outputs)
            if return_outputs
            else total_loss
        )

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        inputs.pop("kd_id", None)
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self,data_args:"DataArguments", dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)
        decoded_preds = [pred.strip().lower() for pred in decoded_preds]
        decoded_labels = [label.strip().lower() for label in decoded_labels]
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

        output_f1_score_file = os.path.join(self.args.output_dir, "generated_predictions_f1.jsonl")
        # decoded_preds = [match_emotion(data_args.eval_dataset, each) for each in decoded_preds]
        score, res_matrix = report_score(golds=decoded_labels, preds=decoded_preds)
        with open(output_f1_score_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(score))
            f.write(f'\n{res_matrix}')