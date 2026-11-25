# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen
import numpy as np
import os
if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    #############################
    def __post_init__(self):
        # 1. 极其关键的安全防御：如果父类也有 post_init，先执行父类的，防止框架崩溃
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # 2. 执行我们自己的向量懒加载逻辑
        # self.vk_vectors_dict = {}
        # if getattr(self.data_args, "vk_vector_path", None):
        #     try:
        #         self.vk_vectors_dict = np.load(self.data_args.vk_vector_path, allow_pickle=True).item()
        #         logger.info_rank0(f" 成功从 {self.data_args.vk_vector_path} 加载了 {len(self.vk_vectors_dict)} 个知识向量")
        #     except Exception as e:
        #         logger.warning_rank0(f" 知识向量加载失败，路径: {self.data_args.vk_vector_path}, 错误: {e}")
        # self.vk_vectors_dict = {}
        # if getattr(self.data_args, "vk_vector_path", None):
        #     try:
        #         # [修改点 1] 直接加载 .npz 文件，彻底抛弃危险的 allow_pickle=True
        #         npz_data = np.load(self.data_args.vk_vector_path)
                
        #         # [修改点 2] 强转为普通 dict，断开与磁盘文件的物理连接，确保 DataLoader 多进程安全！
        #         self.vk_vectors_dict = dict(npz_data)
                
        #         # len() 现在可以直接作用于普通的字典上了
        #         logger.info_rank0(f" 成功从 {self.data_args.vk_vector_path} 加载了 {len(self.vk_vectors_dict)} 个知识向量")
        #     except Exception as e:
        #         logger.warning_rank0(f" 知识向量加载失败，路径: {self.data_args.vk_vector_path}, 错误: {e}")
        # self.vk_vectors_dict = {}
        # if getattr(self.data_args, "vk_vector_path", None):
        #     # 现在要求 yaml 里传入的 vk_vector_path 是这些 npz 所在的【文件夹路径】
        #     base_dir = self.data_args.vk_vector_path
        #     layer_mapping = {
        #         # "layer_20": "dynamic_70b_hidden_states_20",
        #         # "layer_30": "dynamic_70b_hidden_states_30",
        #         # "layer_40": "dynamic_70b_hidden_states_40",
        #         "layer_50": "dynamic_70b_hidden_states_50",
        #         "layer_60": "dynamic_70b_hidden_states_60",
        #         "layer_70": "dynamic_70b_hidden_states_70",
        #         # "layer_80": "dynamic_70b_hidden_states_80"
        #     }
        #     for internal_name, file_prefix in layer_mapping.items():
        #         # 拼接文件路径
        #         file_path = os.path.join(base_dir, f"{file_prefix}.npz")
                
        #         try:
        #             npz_data = np.load(file_path)
                    
        #             # 强转为 dict，强制将 Key 转为 str，并且使用 internal_name (例如 "layer_20") 作为字典的 Key
        #             self.vk_vectors_dict[internal_name] = {str(k): v for k, v in npz_data.items()}
                    
        #             logger.info_rank0(f"✅ 成功从 {file_path} 加载了 {len(self.vk_vectors_dict[internal_name])} 个 {internal_name} 知识向量")
        #         except Exception as e:
        #             logger.warning_rank0(f"❌ 知识向量加载失败，缺少层: {internal_name}, 路径: {file_path}, 错误: {e}")
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
        #########################################################
            model_inputs["kd_id"].append(int(examples["id"][i]))
        ###########################################################
            # if getattr(self.data_args, "vk_vector_path", None) is not None:
            #     if "id" in examples:
            #         current_id = str(examples["id"][i])
            #         # print(self.vk_vectors_dict[0])
            #         if current_id in self.vk_vectors_dict:
            #             model_inputs["target_vk_vectors"].append(self.vk_vectors_dict[current_id].tolist())
            #         else:
            #             model_inputs["target_vk_vectors"].append([0.0] * 768)
            # if getattr(self.data_args, "vk_vector_path", None) is not None:
            #     if "id" in examples:
            #         current_id = str(examples["id"][i])
            #         internal_layer_names = ["layer_50", "layer_60","layer_70"]
                    
            #         for internal_name in internal_layer_names:
            #             # 动态生成独立的列名，比如 "target_vk_layer_20"
            #             col_name = f"target_vk_{internal_name}"
                        
            #             if internal_name in self.vk_vectors_dict and current_id in self.vk_vectors_dict[internal_name]:
            #                 model_inputs[col_name].append(self.vk_vectors_dict[internal_name][current_id].tolist())
            #             else:
            #                 model_inputs[col_name].append([0.0] * 8192)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
