# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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
from sklearn.metrics import f1_score,accuracy_score
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available
import re

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)

@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))
        if compute_result:
            return self._dump()



# @dataclass
# class ComputeAccuracy:
#     r"""Compute accuracy and support `batch_eval_metrics`."""

#     def _dump(self) -> Optional[dict[str, float]]:
#         result = None
#         if hasattr(self, "score_dict"):
#             result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
#         self.score_dict = {"accuracy": [],"F1":[]}
#         return result

#     def __post_init__(self):
#         self._dump()

#     def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
#         preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
#         for i in range(len(preds)):
#             pred, label = preds[i, :-1], labels[i, 1:]
#             label_mask = label != IGNORE_INDEX
#             self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))
#         preds, labels = preds[:,:-1], labels[:,1:]    
#         labels_mask = labels != IGNORE_INDEX
#         self.score_dict["F1"] = f1_score(labels[labels_mask],preds[labels_mask],average='macro')
#         if compute_result:
#             return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": [],"F1":[],"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        #进行魔改
        # self.score_dict["F1"] = f1_score(decoded_labels,decoded_preds,average='macro')
        # self.score_dict["accuracy"] = accuracy_score(decoded_labels,decoded_preds)
        clean_preds = []
        clean_labels = []

        # 遍历解码后的每一对预测和标签
        for p, l in zip(decoded_preds, decoded_labels):
            # 正则强行抠出三个立场词（防止前后有空格或标点符号干扰）
            p_match = re.search(r'\b(favor|against|none)\b', str(p), re.IGNORECASE)
            p_stance = p_match.group(1) if p_match else "none"
            
            l_match = re.search(r'\b(favor|against|none)\b', str(l), re.IGNORECASE)
            l_stance = l_match.group(1) if l_match else "none"

            clean_preds.append(p_stance)
            clean_labels.append(l_stance)

        # 1. 计算三分类的总体准确率 (如果你需要看总体准度的话)
        self.score_dict["accuracy"] = accuracy_score(clean_labels, clean_preds)

        # 2. 严格按照 MT-CSD 学术标准，只算 favor 和 against 的 Macro-F1
        try:
            f1_fa = f1_score(
                y_true=clean_labels, 
                y_pred=clean_preds, 
                labels=["favor", "against"],  # 强制锁定这两类，剔除 None
                average="macro"
            )
        except Exception as e:
            f1_fa = 0.0
            print(f" 文本 F1 计算出错: {e}")

        # 覆盖你字典里的 F1
        self.score_dict["F1"] = f1_fa
        if compute_result:
            return self._dump()

#################################################################################################################
# @dataclass
# class ComputeSimilarity:
#     r"""Compute text similarity scores and support `batch_eval_metrics`.

#     Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
#     """

#     tokenizer: "PreTrainedTokenizer"

#     def _dump(self) -> Optional[dict[str, float]]:
#         result = None
#         if hasattr(self, "score_dict"):
#             # 1. 传统的 ROUGE 和 BLEU 依然使用列表平均
#             result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

#             # 2. === 全局精准计算 Accuracy 和 F1 ===
#             # 只有当收集到了完整的标签时才计算，避免除零错误
#             if hasattr(self, "all_clean_labels") and len(self.all_clean_labels) > 0:
#                 acc = accuracy_score(self.all_clean_labels, self.all_clean_preds) * 100
#                 try:
#                     f1_fa = f1_score(
#                         y_true=self.all_clean_labels, 
#                         y_pred=self.all_clean_preds, 
#                         labels=["Favor", "Against"],  # 强制锁定这两类，剔除 None
#                         average="macro"
#                     ) * 100
#                 except Exception as e:
#                     f1_fa = 0.0
#                     print(f"文本 F1 计算出错: {e}")
                
#                 # 直接将全局结果写入返回的字典中
#                 result["accuracy"] = float(acc)
#                 result["F1"] = float(f1_fa)

#         # 3. 清空字典和全局列表，为下一个 Epoch 做准备
#         self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
#         self.all_clean_preds = []
#         self.all_clean_labels = []
        
#         return result

#     def __post_init__(self):
#         # 初始化全局变量
#         self.all_clean_preds = []
#         self.all_clean_labels = []
#         self._dump()

#     def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
#         preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

#         preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
#         labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

#         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

#         # --- 原始 ROUGE / BLEU 计算 (保持不变) ---
#         for pred, label in zip(decoded_preds, decoded_labels):
#             hypothesis = list(jieba.cut(pred))
#             reference = list(jieba.cut(label))

#             if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
#                 result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
#             else:
#                 rouge = Rouge()
#                 scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
#                 result = scores[0]

#             for k, v in result.items():
#                 self.score_dict[k].append(round(v["f"] * 100, 4))

#             bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
#             self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

#         # --- 核心魔改区：收集本 Batch 的标签 ---
#         for p, l in zip(decoded_preds, decoded_labels):
#             p_str = str(p)
#             l_str = str(l).strip()

#             # 提取预测值
#             match = re.search(r"<Stance>(.*?)</Stance>", p_str, re.IGNORECASE)
#             if match:
#                 p_stance = match.group(1).strip().capitalize()
#             else:
#                 if "Favor" in p_str or "favor" in p_str:
#                     p_stance = "Favor"
#                 elif "Against" in p_str or "against" in p_str:
#                     p_stance = "Against"
#                 else:
#                     p_stance = "None"
            
#             # 提取真实值
#             l_stance = l_str.capitalize()
#             if l_stance not in ["Favor", "Against", "None"]:
#                 l_match = re.search(r'\b(Favor|Against|None)\b', l_str, re.IGNORECASE)
#                 l_stance = l_match.group(1).capitalize() if l_match else "None"

#             # 🚨 重点：只添加到全局列表，本回合不计算 F1
#             self.all_clean_preds.append(p_stance)
#             self.all_clean_labels.append(l_stance)

#         # 如果 compute_result=True，说明所有 Batch 已经跑完，触发 _dump 进行最终计算
#         if compute_result:
#             return self._dump()