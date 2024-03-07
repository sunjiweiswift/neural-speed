//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/data_types.h"
#include "core/ne.h"
#include "core/ne_layers.h"
#include "models/falcon/falcon.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/quant_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

void model_load_internal(const std::string& fname, model_archs arch, model_context* ctx, int n_gpu_layers,
                         bool use_mmap, bool use_mlock, bool vocab_only, model_progress_callback progress_callback,
                         void* progress_callback_user_data) {
  std::unique_ptr<FALCON> ms(new FALCON());
  ms->init(fname.c_str(), ctx, n_gpu_layers, use_mmap, use_mlock, vocab_only);
  ms->load(ctx, progress_callback, progress_callback_user_data);
  model_context& lctx = *ctx;
  lctx.support_bestla_kv = true;
}

void FALCON::init(const char* path_model, model_context* ctx, int n_gpu_layer_, bool use_mmap_, bool use_mlock_,
                  bool vocab_only_) {
  model_context& lctx = *ctx;
  n_gpu_layer = n_gpu_layer_;
  use_mmap = use_mmap_;
  use_mlock = use_mlock_;
  vocab_only = vocab_only_;
  auto& model = lctx.model;
  ml.reset(new model_model_loader(path_model, use_mmap, vocab_only));
  lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
  model.hparams = ml->file_loaders.at(0)->hparams;
  model_file_version file_version = ml->file_loaders.at(0)->file_version;
  auto& hparams = model.hparams;
  n_ff = 4 * hparams.n_embd;
  fprintf(stderr, "%s: n_vocab    = %u\n", __func__, hparams.n_vocab);
  fprintf(stderr, "%s: n_ctx      = %u\n", __func__, hparams.max_seq_len);
  fprintf(stderr, "%s: n_embd     = %u\n", __func__, hparams.n_embd);
  fprintf(stderr, "%s: n_mult     = %u\n", __func__, hparams.n_mult);
  fprintf(stderr, "%s: n_head     = %u\n", __func__, hparams.n_head);
  fprintf(stderr, "%s: n_head_kv  = %u\n", __func__, hparams.n_head_kv);
  fprintf(stderr, "%s: n_layer    = %u\n", __func__, hparams.n_layer);
  fprintf(stderr, "%s: n_rot      = %u\n", __func__, hparams.n_rot);
  fprintf(stderr, "%s: n_ff       = %u\n", __func__, n_ff);
  fprintf(stderr, "%s: n_parts    = %zu\n", __func__, ml->file_loaders.size());
  n_embd = hparams.n_embd;
  n_vocab = hparams.n_vocab;
  n_layer = hparams.n_layer;
  n_head_kv = hparams.n_head_kv;
  scratch = falcon_mem_req(n_layer, lctx.model_scratch_size_ratio);
  model.scratchs = scratch;
}

#define MODEL_BACKEND_OFFLOAD NE_BACKEND_CPU
void FALCON::load(model_context* ctx, model_progress_callback progress_callback, void* progress_callback_user_data) {
  model_context& lctx = *ctx;
  auto& model = lctx.model;
  auto& ne_ctx = model.ctx;

  size_t ctx_size;
  size_t mmapped_size;
  ml->calc_sizes(&ctx_size, &mmapped_size);
  fprintf(stderr, "%s: ne ctx size = %7.2f MB\n", __func__, ctx_size / 1024.0 / 1024.0);

  // create the ne context
  lctx.model.buf.resize(ctx_size);
  if (use_mlock) {
    lctx.model.mlock_buf.init(lctx.model.buf.addr);
    lctx.model.mlock_buf.grow_to(lctx.model.buf.size);
  }

  struct ne_init_params params = {
      /*.mem_size   =*/lctx.model.buf.size,
      /*.mem_buffer =*/lctx.model.buf.addr,
      /*.no_alloc   =*/ml->use_mmap,
  };

  model.ctx = ne_init(params);
  if (!model.ctx) {
    throw format("ne_init() failed");
  }

  ml->ne_ctx = ne_ctx;
  const int i_gpu_start = n_layer - n_gpu_layer;

  model.layers.resize(n_layer);
  size_t vram_total = 0;

  if (ml->verify_tensor("transformer.word_embeddings.weight")) {
    model.others[0] = ml->get_tensor("transformer.word_embeddings.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
    model.others[1] = ml->get_tensor("transformer.ln_f.weight", {n_embd}, NE_BACKEND_CPU);
    model.others[2] = ml->get_tensor("transformer.ln_f.bias", {n_embd}, NE_BACKEND_CPU);
    model.others[3] = ml->get_tensor("lm_head.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);

    for (uint32_t i = 0; i < n_layer; ++i) {
      const ne_backend backend = static_cast<int>(i) < i_gpu_start ? NE_BACKEND_CPU : MODEL_BACKEND_OFFLOAD;
      auto& layer = model.layers[i];
      std::string layers_i = "transformer.h." + std::to_string(i);

      // norm: cur = ln_1_g*cur + ln_1_b
      if (n_head_kv == 1) {  //  7B
        layer.norm[0] = ml->get_tensor(layers_i + ".input_layernorm.weight", {n_embd}, backend);
        layer.norm[1] = ml->get_tensor(layers_i + ".input_layernorm.bias", {n_embd}, backend);
      } else if (n_head_kv == 8) {  // 40B
        layer.norm[0] = ml->get_tensor(layers_i + ".ln_mlp.weight", {n_embd}, backend);
        layer.norm[1] = ml->get_tensor(layers_i + ".ln_mlp.bias", {n_embd}, backend);
        layer.norm[2] = ml->get_tensor(layers_i + ".ln_attn.weight", {n_embd}, backend);
        layer.norm[3] = ml->get_tensor(layers_i + ".ln_attn.bias", {n_embd}, backend);
      } else {
        fprintf(stderr, "n_head_kv should be 1 (7B) or 8 (40B) in Falcon model, rather %d \n", n_head_kv);
      }

      // qkv GEMM
      layer.attn[0] = ml->get_tensor(layers_i + ".self_attention.query_key_value.weight",
                                     {n_embd, n_embd + 2 * n_head_kv * (n_embd / model.hparams.n_head)}, backend);
      layer.attn[1] = ml->get_tensor(layers_i + ".self_attention.dense.weight", {n_embd, n_embd}, backend);

      // ffn GEMM
      layer.ffn[0] = ml->get_tensor(layers_i + ".mlp.dense_h_to_4h.weight", {n_embd, n_ff}, backend);
      layer.ffn[1] = ml->get_tensor(layers_i + ".mlp.dense_4h_to_h.weight", {n_ff, n_embd}, backend);

      if (backend != NE_BACKEND_CPU) {
        vram_total += ne_nbytes(layer.norm[0]) + ne_nbytes(layer.norm[1]) + ne_nbytes(layer.attn[0]) +
                      ne_nbytes(layer.attn[1]) + ne_nbytes(layer.ffn[0]) + ne_nbytes(layer.ffn[1]);
        if (n_head_kv == 8) {
          vram_total += ne_nbytes(layer.norm[2]) + ne_nbytes(layer.norm[3]);
        }
      }
    }
  } else {
    model.others[0] = ml->get_tensor("token_embd.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
    model.others[1] = ml->get_tensor("output_norm.weight", {n_embd}, NE_BACKEND_CPU);
    model.others[2] = ml->get_tensor("output_norm.bias", {n_embd}, NE_BACKEND_CPU);
    model.others[3] = ml->get_tensor("output.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);

    for (uint32_t i = 0; i < n_layer; ++i) {
      const ne_backend backend = static_cast<int>(i) < i_gpu_start ? NE_BACKEND_CPU : MODEL_BACKEND_OFFLOAD;
      auto& layer = model.layers[i];
      std::string layers_i = "blk." + std::to_string(i);

      // norm: cur = ln_1_g*cur + ln_1_b
      if (n_head_kv == 1) {  //  7B
        layer.norm[0] = ml->get_tensor(layers_i + ".attn_norm.weight", {n_embd}, backend);
        layer.norm[1] = ml->get_tensor(layers_i + ".attn_norm.bias", {n_embd}, backend);
      } else if (n_head_kv == 8) {  // 40B
        layer.norm[0] = ml->get_tensor(layers_i + ".attn_norm.weight", {n_embd}, backend);
        layer.norm[1] = ml->get_tensor(layers_i + ".attn_norm.bias", {n_embd}, backend);
        layer.norm[2] = ml->get_tensor(layers_i + ".attn_norm_2.weight", {n_embd}, backend);
        layer.norm[3] = ml->get_tensor(layers_i + ".attn_norm_2.bias", {n_embd}, backend);
      } else {
        fprintf(stderr, "n_head_kv should be 1 (7B) or 8 (40B) in Falcon model, rather %d \n", n_head_kv);
      }

      // qkv GEMM
      layer.attn[0] = ml->get_tensor(layers_i + ".attn_qkv.weight",
                                     {n_embd, n_embd + 2 * n_head_kv * (n_embd / model.hparams.n_head)}, backend);
      layer.attn[1] = ml->get_tensor(layers_i + ".attn_output.weight", {n_embd, n_embd}, backend);

      // ffn GEMM
      layer.ffn[0] = ml->get_tensor(layers_i + ".ffn_up.weight", {n_embd, n_ff}, backend);
      layer.ffn[1] = ml->get_tensor(layers_i + ".ffn_down.weight", {n_ff, n_embd}, backend);

      if (backend != NE_BACKEND_CPU) {
        vram_total += ne_nbytes(layer.norm[0]) + ne_nbytes(layer.norm[1]) + ne_nbytes(layer.attn[0]) +
                      ne_nbytes(layer.attn[1]) + ne_nbytes(layer.ffn[0]) + ne_nbytes(layer.ffn[1]);
        if (n_head_kv == 8) {
          vram_total += ne_nbytes(layer.norm[2]) + ne_nbytes(layer.norm[3]);
        }
      }
    }
  }

  // print memory requirements
  // this is the total memory required to run the inference
  const size_t mem_required = ctx_size + mmapped_size - vram_total +  // weights in VRAM not in memory
                              scratch.scratch0 + scratch.scratch1 + scratch.eval;
  fprintf(stderr, "%s: mem required  = %7.2f MB (+ memory per state)\n", __func__, mem_required / 1024.0 / 1024.0);

  (void)n_gpu_layer;

  // populate `tensors_by_name`
  for (model_load_tensor& lt : ml->tensors_map.tensors) {
    model.tensors_by_name.emplace_back(lt.name, lt.ne_tensor);
  }

  ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &lctx.model.mlock_mmap : nullptr);

  if (progress_callback) {
    progress_callback(1.0f, progress_callback_user_data);
  }

  model.mapping = std::move(ml->mapping);
}

#undef MODEL_BACKEND_OFFLOAD

class falcon_quant_layer : public quant_layer_base {
 public:
  quant_params_internal get_layer_config(std::string layername, std::vector<int64_t> ne, ne_type type) override {
    bool quantize = layername.rfind("weight") == layername.size() - 6;
    if (layername == "transformer.word_embeddings.weight" || layername == "token_embd.weight") {
      // special layer process, can be loaded by config file
      return quant_params_internal();  // return q4_0 to cover the usage of getrow
    }
    quantize &= (ne.size() == 2);
    if (quantize) {
      return mGCfg;  // use global quant config
    } else {
      return quant_params_internal{quant_bits::count};  // non-quant
    }
  }
};
REGISTER_QUANT_LAYER_CLASS(falcon);
