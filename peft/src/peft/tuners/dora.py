# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Dora`].

    Args:
        r (`int`): Dora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Dora to.
        dora_alpha (`float`): The alpha parameter for Dora scaling.
        dora_dropout (`float`): The dropout probability for Dora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Dora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_dora ( `List[bool]`): Used with `dora.MergedLinear`.
        bias (`str`): Bias type for Dora. Can be 'none', 'all' or 'dora_only'
        modules_to_save (`List[str]`):List of modules apart from DoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Dora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Dora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    dora_alpha: int = field(default=None, metadata={"help": "Dora alpha"})
    dora_dropout: float = field(default=None, metadata={"help": "Dora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Dora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_dora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `dora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Dora. Can be 'none', 'all' or 'dora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from DoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.DORA


class DoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Dora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`DoraConfig`]): The configuration of the Dora model.

    Returns:
        `torch.nn.Module`: The Dora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, DoraConfig >>> from peft import DoraModel, DoraConfig >>>
        config = DoraConfig(
            peft_type="DORA", task_type="SEQ_2_SEQ_LM", r=8, dora_alpha=32, target_modules=["q", "v"],
            dora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> dora_model = DoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`DoraConfig`]): The configuration of the Dora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_dora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Dora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "dora_alpha": self.peft_config.dora_alpha,
            "dora_dropout": self.peft_config.dora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_dora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_dora": self.peft_config.enable_dora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_dora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_dora is not None:
                    kwargs.update({"enable_dora": self.peft_config.enable_dora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "dora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, DoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/DoRA/blob/main/doralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `dora_only` to work
def mark_only_dora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "dora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "dora_only":
        for m in model.modules():
            if isinstance(m, DoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class DoraLayer:
    def __init__(
        self,
        r: int,
        dora_alpha: int,
        dora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.dora_alpha = dora_alpha
        # Optional dropout
        if dora_dropout > 0.0:
            self.dora_dropout = nn.Dropout(p=dora_dropout)
        else:
            self.dora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False
        self._caches: dict[str, Any] = {}


class Linear(nn.Linear, DoraLayer):
    # Dora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        dora_alpha: int = 1,
        dora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DoraLayer.__init__(self, r=r, dora_alpha=dora_alpha, dora_dropout=dora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.dora_A = nn.Linear(in_features, r, bias=False)
            self.dora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.dora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            
        self.reset_parameters()
        if r > 0:
            dora_weight = self.dora_B.weight @ self.dora_A.weight
            weight_norm = self._get_weight_norm(self.weight, dora_weight, self.scaling)
            # magnitude column-wise across output dimension
            self.magnitude = nn.Parameter(weight_norm, requires_grad=True)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "dora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.dora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.dora_B.weight)

    def _get_weight_norm(self, weight, dora_weight, scaling):
        weight = weight + scaling * dora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
    
    def _apply_dora(self, x, dora_A, dora_B, scaling):
        dora_weight = dora_B.weight @ dora_A.weight
        weight_norm = self._get_weight_norm(self.weight, dora_weight, scaling).detach()
        mag_norm_scale = (self.magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * F.linear(x, self.weight, self.bias) + mag_norm_scale * dora_B(dora_A(x)) * scaling
        return result_dora

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.dora_A.train(mode)
        self.dora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                delta_weight = transpose(self.dora_B.weight @ self.dora_A.weight, self.fan_in_fan_out) * self.scaling
                weight_norm = self._get_weight_norm(
                            self.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                        ).detach()
                self._cache_store(f"weight_norm", weight_norm)
                dora_factor = self.magnitude / weight_norm
                dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                new_weight = dora_factor * (self.weight.data + delta_weight)
                self.weight.data = new_weight
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                delta_weight = transpose(self.dora_B.weight @ self.dora_A.weight, self.fan_in_fan_out) * self.scaling
                weight_norm = self._cache_pop(f"weight_norm")
                dora_factor = self.magnitude / weight_norm
                weight_orig = self.weight.data / dora_factor.view(-1, 1) - delta_weight
                self.weight.data = weight_orig
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.dora_A.eval()
        self.dora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            if self.r > 0 and self.merged:
                delta_weight = transpose((self.dora_B.weight @ self.dora_A.weight).to(previous_dtype), self.fan_in_fan_out) * self.scaling
                weight_norm = self._cache_pop(f"weight_norm")
                dora_factor = self.magnitude / weight_norm
                weight_orig = self.weight.data / dora_factor.view(-1, 1) - delta_weight
                self.weight.data = weight_orig
                self.merged = False
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                x = self.dora_dropout(x)
                result += self._apply_dora(x, self.dora_A, self.dora_B, self.scaling)
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class MergedLinear(nn.Linear, DoraLayer):
    # Dora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        dora_alpha: int = 1,
        dora_dropout: float = 0.0,
        enable_dora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DoraLayer.__init__(self, r=r, dora_alpha=dora_alpha, dora_dropout=dora_dropout, merge_weights=merge_weights)
        if out_features % len(enable_dora) != 0:
            raise ValueError("The length of enable_dora must divide out_features")
        self.enable_dora = enable_dora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_dora):
            self.dora_A = nn.Linear(in_features, r * sum(enable_dora), bias=False)
            self.dora_B = nn.Conv1d(
                r * sum(enable_dora),
                out_features // len(enable_dora) * sum(enable_dora),
                kernel_size=1,
                groups=2,
                bias=False,
            )
            self.scaling = self.dora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.dora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_dora), -1)
            self.dora_ind[enable_dora, :] = True
            self.dora_ind = self.dora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "dora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.dora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.dora_B.weight)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.dora_ind] = x.reshape(-1, self.out_features // len(self.enable_dora) * sum(self.enable_dora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.dora_A.train(mode)
        self.dora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_dora):
                delta_w = (
                    F.conv1d(
                        self.dora_A.weight.data.unsqueeze(0),
                        self.dora_B.weight.data,
                        groups=sum(self.enable_dora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data += transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_dora):
                delta_w = (
                    F.conv1d(
                        self.dora_A.weight.data.unsqueeze(0),
                        self.dora_B.weight.data,
                        groups=sum(self.enable_dora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.dora_A.eval()
        self.dora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.r > 0 and self.merged and any(self.enable_dora):
                delta_w = (
                    F.conv1d(
                        self.dora_A.weight.data.unsqueeze(0),
                        self.dora_B.weight.data,
                        groups=sum(self.enable_dora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                delta_w = delta_w.to(self.weight.dtype)
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
                self.merged = False
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                after_A = self.dora_A(self.dora_dropout(x.to(self.dora_A.weight.dtype)))
                after_B = self.dora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, DoraLayer):
        # Dora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            dora_alpha: int = 1,
            dora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            DoraLayer.__init__(self, r=r, dora_alpha=dora_alpha, dora_dropout=dora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                self.dora_A = nn.Linear(in_features, r, bias=False)
                self.dora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.dora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "dora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.dora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.dora_B.weight)

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = self.dora_B(self.dora_A(self.dora_dropout(x))).to(expected_dtype) * self.scaling
                    result += output
                else:
                    output = self.dora_B(self.dora_A(self.dora_dropout(x))) * self.scaling
                    result += output
            return result

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, DoraLayer):
        # Dora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            dora_alpha: int = 1,
            dora_dropout: float = 0.0,
            enable_dora: List[bool] = [False],
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            DoraLayer.__init__(self, r=r, dora_alpha=dora_alpha, dora_dropout=dora_dropout, merge_weights=False)
            if out_features % len(enable_dora) != 0:
                raise ValueError("The length of enable_dora must divide out_features")
            self.enable_dora = enable_dora
            # Actual trainable parameters
            if r > 0 and any(enable_dora):
                self.dora_A = nn.Linear(in_features, r * sum(enable_dora), bias=False)
                self.dora_B = nn.Conv1d(
                    r * sum(enable_dora),
                    out_features // len(enable_dora) * sum(enable_dora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                )
                self.scaling = self.dora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                # Compute the indices
                self.dora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_dora), -1)
                self.dora_ind[enable_dora, :] = True
                self.dora_ind = self.dora_ind.view(-1)
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "dora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.dora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.dora_B.weight)

        def zero_pad(self, x):
            result = x.new_zeros((*x.shape[:-1], self.out_features))
            result = result.view(-1, self.out_features)
            result[:, self.dora_ind] = x.reshape(
                -1, self.out_features // len(self.enable_dora) * sum(self.enable_dora)
            )
            return result.view((*x.shape[:-1], self.out_features))

        def forward(self, x: torch.Tensor):
            result = super().forward(x)
            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()
                    after_A = self.dora_A(self.dora_dropout(x))
                    after_B = self.dora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                    result += output
                else:
                    after_A = self.dora_A(self.dora_dropout(x))
                    after_B = self.dora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B) * self.scaling
                    result += output
            return result
