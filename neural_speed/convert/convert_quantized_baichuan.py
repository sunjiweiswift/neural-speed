#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import sys
import re
import argparse
from common import *
from sentencepiece import SentencePieceProcessor


def load_vocab_for_baichuan(path: Path) -> SentencePieceVocab:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            raise FileNotFoundError(
                f"Could not find tokenizer.model in {path} or its parent; if it's in another directory, \
                pass the directory as --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)


def convert_to_qx_bestla_tensor(src_name, dst_name, model, fout, q_config):
    # unpack weight and repack into 3bits / 4bits BestLA format
    import neural_speed.llama_cpp as cpp_model
    if ".weight" in src_name:
        src_name = src_name.replace(".weight", "")
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]

    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    int_weight = int_weight.view(-1, int_weight.shape[-1])

    # shuffle weight in GPTQ when act order is on
    if 'desc_act' in q_config and q_config['desc_act']:
        g_idx = model[f"{src_name}.g_idx"]
        int_weight2 = int_weight.clone()
        group_size = q_config['group_size']
        group_dict = {}
        for i in range(len(g_idx)):
            group_idx = g_idx[i].item()
            if group_idx not in group_dict:
                target_idx = group_idx * group_size
                group_dict[group_idx] = 0
            else:
                group_dict[group_idx] = group_dict[group_idx] + 1
                target_idx = group_idx * group_size + group_dict[group_idx]
            int_weight2[target_idx] = int_weight[i]
        int_weight = int_weight2

    # shape = int_weight.shape[::-1]
    shape = int_weight.shape[::-1]
    # write_header(fout, shape[::-1], dst_name, GGML_QJBLAS_TYPE)
    n_dims = len(shape)
    str = dst_name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), GGML_QJBLAS_TYPE))
    for i in range(n_dims):
        fout.write(struct.pack("i", shape[n_dims - 1 - i]))
    fout.write(str)

    # INC stores sig-int4 value as u4(range 0~15, they add a offset),
    # BesTLA requires s4_clip((-8,7)*16), so we sub the offset and then mul 16.
    # Int3 is the same as int4, but offset=4, mul scale==32.
    weight_dtype = "int8"
    if q_config['bits'] == 4:
        int_weight = (int_weight - 8) * 16
        gptq_scales = gptq_scales / 16
        gptq_zeros = (gptq_zeros - 8) * 16
        weight_dtype = "int4"
    elif q_config['bits'] == 3:
        int_weight = (int_weight - 4) * 32
        gptq_scales = gptq_scales / 32
        gptq_zeros = (gptq_zeros - 4) * 32
        weight_dtype = "int3"
    else:
        ValueError(f"Unsupported q_config[bits]: {q_config['bits']}")

    dst = np.zeros((int_weight.shape[0], int_weight.shape[1] * 4), dtype=np.int8)
    int_weight = np.ascontiguousarray(int_weight.numpy())
    gptq_scales = np.ascontiguousarray((gptq_scales.float()).numpy())
    if q_config['sym']:
        gptq_zeros = np.empty(0, dtype=np.int8)
    else:
        gptq_zeros = np.ascontiguousarray(gptq_zeros.numpy())
    if 'desc_act' in q_config and q_config['desc_act']:
        g_idx = np.ascontiguousarray(g_idx.numpy())
    else:
        g_idx = np.empty(0, dtype=np.int32)

    # repack int weight in BesTLA format
    byte_size = cpp_model.Model.np_bestla_qpack(int_weight,
                                                gptq_scales,
                                                gptq_zeros,
                                                g_idx,
                                                dst,
                                                weight_dtype=weight_dtype,
                                                group_size=q_config['group_size'],
                                                alg="sym" if q_config['sym'] else "asym",
                                                compute_dtype="int8")
    dst.flatten()[:byte_size].tofile(fout)
    print(f"convert_to_qx_bestla_tensor: {src_name:>40} -> {dst_name:<40} shape: {shape}, byte_size: {byte_size:<10}")


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("--model_hub",
                        choices=["huggingface", "modelscope"],
                        default="huggingface",
                        help="hub to load model")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # QWEN-GPTQ & AWQ
    model, hparams, quantize_config = load_quantized_safetensors(model_path)
    list_vars = model

    print(hparams)

    # orinal QWEN
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # hparams = model.config.to_dict()
    # list_vars = model.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    fout = open(out_path, "wb")

    # possible data types
    #   ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0
    if args.outtype == "f16":
        ftype = 1

    # 1. write hparams
    # 0x67676d6c is unversioned ne
    # 0x67676d66 is versioned ggmf (requires token scores)
    ne_file_magic = 0x67676d66
    #ne_file_version = 0x00000001 # v1

    print(hparams)

    fout.write(struct.pack("i", 0x67676d66))
    fout.write(struct.pack("i", 1))

    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", hparams["num_attention_heads"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", hparams["num_hidden_layers"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", ftype))
    fout.write(struct.pack("i", hparams["model_max_length"]))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("i", 0))

    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", hparams["intermediate_size"]))
    fout.write(struct.pack("i", 0))  # n_experts
    fout.write(struct.pack("i", 0))  # n_expert_used
    fout.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor

    fout.write(struct.pack("f", 0.0))  # config.json "rope_scaling.factor", not enabled
    fout.write(struct.pack("i", 0))  # rope_scaling.original_max_position_embeddings
    fout.write(struct.pack("i", 0))  # params["rope_scaling"]["type"] =="yarn" else 0))

    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    # 2. vocab
    tokenizer_path = Path(tokenizer.vocab_file).parent
    vocab = load_vocab_for_baichuan(Path(tokenizer_path))
    counter = 0
    for text, score in vocab.all_tokens():
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        fout.write(struct.pack("f", score))
        counter += 1

    while counter < hparams["vocab_size"]:
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        fout.write(struct.pack("f", 0))
        counter += 1

    def convert_qwen_to_fp32_tensor(src_name, dst_name, model, fout):
        # qwen-gptq is torch.bfloat16 mostly.
        if model[src_name].dtype == torch.float32:
            data = model[src_name].squeeze().numpy()
        else:
            data = model[src_name].squeeze().to(torch.float32).numpy()
        data = data.astype(np.float32)
        shape = data.shape
        n_dims = len(shape)
        print("convert_qwen_to_fp32_tensor:  %40s" % src_name + "-> %-40s" % dst_name + " shape: ", shape, " type: ",
              data.dtype)

        #ftype_cur = {torch.float16: 1, torch.float32: 0}[data.dtype]
        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            data = data.astype(np.float32)

        # header
        # write_header(fout, shape, dst_name, ftype_cur)
        str = src_name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    for name in list_vars.keys():
        print(name)

    #3. write tensors
    convert_qwen_to_fp32_tensor("model.embed_tokens.weight", "model.embed_tokens.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("model.norm.weight", "model.norm.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.weight", "lm_head.weight", list_vars, fout)

    for i in range(hparams["num_hidden_layers"]):
        prefix = "model.layers." + str(i)

        convert_qwen_to_fp32_tensor(f"{prefix}.input_layernorm.weight", f"{prefix}.input_layernorm.weight", list_vars,
                                    fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.post_attention_layernorm.weight",
                                    f"{prefix}.post_attention_layernorm.weight", list_vars, fout)
        # qkv GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.self_attn.W_pack.weight", f"{prefix}.self_attn.W_pack.weight", list_vars,
                                    fout, quantize_config)
        convert_qwen_to_fp32_tensor(f"{prefix}.self_attn.W_pack.bias", f"{prefix}.attn.c_attn.bias", list_vars, fout)
        convert_to_qx_bestla_tensor(f"{prefix}.self_attn.o_proj.weight", f"{prefix}.self_attn.o_proj.weight", list_vars,
                                    fout, quantize_config)
        convert_qwen_to_fp32_tensor(f"{prefix}.self_attn.o_proj.bias", f"{prefix}.self_attn.o_proj.bias", list_vars,
                                    fout)

        # ffn GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.gate_proj.weight", list_vars, fout,
                                    quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.down_proj", f"{prefix}.mlp.down_proj.weight", list_vars, fout,
                                    quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.up_proj", f"{prefix}.mlp.up_proj.weight", list_vars, fout,
                                    quantize_config)

    fout.close()
    print(f"Success! saved as {out_path}")


if __name__ == '__main__':
    main()
