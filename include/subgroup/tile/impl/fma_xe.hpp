/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "./blk_mma.hpp"

namespace gpu::xetla::subgroup {

/// @brief Is the tile mma operation functor, specialized for Xe and fpu engine.
template <
    typename matAcc_t_,
    typename matC_t_,
    typename matB_t_,
    typename matA_t_,
    gpu_arch arch_tag_>
struct tile_mma_t<
    matAcc_t_,
    matC_t_,
    matB_t_,
    matA_t_,
    mma_engine::fpu,
    arch_tag_,
    std::enable_if_t<
        arch_has_fpu<arch_tag_> && !matA_t_::reg_transpose &&
        matB_t_::reg_transpose>> {
  using matA_t = matA_t_;
  using matB_t = matB_t_;
  using matC_t = matC_t_;
  using matAcc_t = matAcc_t_;
  using dtype_a = typename matA_t::dtype;
  using dtype_b = typename matB_t::dtype;
  using dtype_c = typename matC_t::dtype;
  using dtype_acc = typename matAcc_t::dtype;

  static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
  static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
  static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
  static constexpr uint32_t a_block_size_y = matA_t::block_size_y;
  static constexpr uint32_t a_block_size_x = matA_t::block_size_x;
  static constexpr uint32_t a_block_elems = matA_t::block_elems;

  static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
  static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
  static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
  static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
  static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
  static constexpr uint32_t b_block_elems = matB_t::block_elems;

  static constexpr uint32_t tile_size_m = a_tile_size_y;
  static constexpr uint32_t tile_size_k = a_tile_size_x;
  static constexpr uint32_t tile_size_n = b_tile_size_x;
  static constexpr uint32_t block_size_m = a_block_size_y;
  static constexpr uint32_t block_size_k = a_block_size_x;
  static constexpr uint32_t block_size_n = b_block_size_x;

  static_assert(
      !matA_t::reg_transpose,
      "For FMAOp GEMM, the register layout of matA should be row-major");
  static_assert(
      matB_t::reg_transpose,
      "For FMAOp GEMM, the register layout of matB should be col-major");

  static_assert(
      a_tile_size_x == b_tile_size_y,
      "matA tile k should match with matB tile k");
  static_assert(
      a_block_size_x == b_block_size_y,
      "matA block k should match with matB block k");
  static_assert(
      b_block_size_x == matC_t::block_size_x,
      "matB block n should match with matAcc block n");
  static_assert(
      a_block_size_y == matC_t::block_size_y,
      "matA block m should match with matAcc block m");

  __XETLA_API static void mma(matAcc_t& acc, matC_t& c, matB_t& b, matA_t& a) {
#pragma unroll
    for (uint32_t k = 0; k < tile_size_k / block_size_k; k++) {
#pragma unroll
      for (uint32_t m = 0; m < tile_size_m / block_size_m; m++) {
        uint32_t a_block_idx = m * tile_size_k / block_size_k + k;
        auto a_block = a.reg.xetla_select<matA_t::block_elems, 1>(
            a_block_idx * matA_t::block_elems);
#pragma unroll
        for (uint32_t n = 0; n < tile_size_n / block_size_n; n++) {
          uint32_t b_block_idx = n * tile_size_k / block_size_k + k;
          auto b_block = b.reg.xetla_select<matB_t::block_elems, 1>(
              b_block_idx * matB_t::block_elems);

          uint32_t c_block_idx = m * tile_size_n / block_size_n + n;
          auto c_block = c.reg.xetla_select<matC_t::block_elems, 1>(
              c_block_idx * matC_t::block_elems);

          acc.reg = 0;
          auto acc_block = acc.reg.xetla_select<matAcc_t::block_elems, 1>(0);
          mma_core<block_size_m, block_size_n, block_size_k>(
              acc_block, b_block, a_block);
          reduce_block_acc_k<block_size_m, block_size_n>(acc_block, c_block);
        }
      }
    }
  }

  template <int blk_m, int blk_n, int blk_k>
  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_acc, blk_m * blk_n * matAcc_t::block_size_x>
          __REF__ acc_block,
      xetla_vector_ref<dtype_b, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_m * blk_k> __REF__ a_block) {
    auto acc_blk_2d = acc_block.xetla_format<
        dtype_acc,
        matAcc_t::block_size_y,
        matAcc_t::block_size_x>();
    auto b_blk_2d = b_block.xetla_format<dtype_b, blk_n, blk_k>();
    auto a_blk_2d = a_block.xetla_format<dtype_a, blk_m, blk_k>();
#pragma unroll
    for (uint32_t m = 0; m < blk_m; m++) {
      auto a_row = a_blk_2d.row(m);
#pragma unroll
      for (uint32_t n = 0; n < blk_n; n++) {
        auto b_row = b_blk_2d.row(n);
        if constexpr (std::is_same_v<dtype_acc, dtype_a>) {
#pragma unroll
          for (uint32_t k = 0; k < blk_k; k += matAcc_t::block_size_x) {
            acc_blk_2d.row(m * blk_n + n) +=
                a_row.xetla_select<matAcc_t::block_size_x, 1>(k) *
                b_row.xetla_select<matAcc_t::block_size_x, 1>(k);
          }
        } else {
          xetla_vector<dtype_a, matAcc_t::block_size_x> acc_tmp = 0;
#pragma unroll
          for (uint32_t k = 0; k < blk_k; k += matAcc_t::block_size_x) {
            acc_tmp += a_row.xetla_select<matAcc_t::block_size_x, 1>(k) *
                b_row.xetla_select<matAcc_t::block_size_x, 1>(k);
          }
          acc_blk_2d.row(m * blk_n + n) = acc_tmp;
        }
      }
    }
  }

  template <int blk_m, int blk_n>
  __XETLA_API static void reduce_block_acc_k(
      xetla_vector_ref<dtype_acc, blk_m * blk_n * matAcc_t::block_size_x>
          __REF__ acc_block,
      xetla_vector_ref<dtype_c, blk_m * blk_n> __REF__ c_block) {
    if constexpr (std::is_same_v<dtype_acc, dtype_c>) {
      c_block += recur_col_reduce<
          reduce_op::sum,
          dtype_acc,
          matAcc_t::block_size_x,
          block_size_m * block_size_n>(acc_block);
    } else {
      auto acc_block_2d = acc_block.xetla_format<
          dtype_acc,
          matAcc_t::block_size_y,
          matAcc_t::block_size_x>();
#pragma unroll
      for (uint32_t mm = 0; mm < block_size_m; mm++) {
#pragma unroll
        for (uint32_t nn = 0; nn < block_size_n; nn++) {
          c_block[mm * block_size_n + nn] += xetla_reduce<
              typename matC_t::dtype,
              dtype_acc,
              matAcc_t::block_size_x,
              reduce_op::sum>(acc_block_2d.row(mm * block_size_n + nn));
        }
      }
    }
  }
};

/// @brief Is the tile mma operation functor, specialized for Xe and fpu
/// engine.
template <
    typename matAcc_dst_t_,
    typename matAcc_src_t_,
    typename matB_t_,
    typename matA_t_,
    gpu_arch arch_tag_>
struct tile_mma_t<
    matAcc_dst_t_,
    matAcc_src_t_,
    matB_t_,
    matA_t_,
    mma_engine::fpu,
    arch_tag_,
    std::enable_if_t<
        arch_has_fpu<arch_tag_> && matA_t_::reg_transpose &&
        !matB_t_::reg_transpose>> {
  using matA_t = matA_t_;
  using matB_t = matB_t_;
  using matSrc_t = matAcc_src_t_;
  using matDst_t = matAcc_dst_t_;
  using dtype_a = typename matA_t::dtype;
  using dtype_b = typename matB_t::dtype;
  using dtype_src = typename matSrc_t::dtype;
  using dtype_dst = typename matDst_t::dtype;

  static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
  static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
  static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
  static constexpr uint32_t a_block_size_y = matA_t::block_size_y;
  static constexpr uint32_t a_block_size_x = matA_t::block_size_x;
  static constexpr uint32_t a_block_elems = matA_t::block_elems;

  static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
  static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
  static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
  static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
  static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
  static constexpr uint32_t b_block_elems = matB_t::block_elems;

  static constexpr uint32_t tile_size_m = matDst_t::tile_size_y;
  static constexpr uint32_t tile_size_k = a_tile_size_x;
  static constexpr uint32_t tile_size_n = matDst_t::tile_size_x;
  static constexpr uint32_t tile_elems = tile_size_m * tile_size_n;
  static constexpr uint32_t block_size_n = matDst_t::block_size_x;
  static constexpr uint32_t block_size_k = a_block_size_x;
  static constexpr uint32_t block_size_m = matDst_t::block_size_y;
  static constexpr uint32_t block_elems = block_size_m * block_size_n;
  static constexpr uint32_t blk_m_iters = tile_size_m / block_size_m;
  static constexpr uint32_t tail_start_m = blk_m_iters * block_size_m;
  static constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
  static constexpr uint32_t tail_start_offset = tail_start_m * tile_size_n;
  static constexpr uint32_t a_tail_start_offset = tail_start_m * a_tile_size_x;

  static_assert(
      matA_t::reg_transpose,
      "For FMAOp GEMM, the register layout of matA should be col-major");
  static_assert(
      !matB_t::reg_transpose,
      "For FMAOp GEMM, the register layout of matB should be row-major");

  static_assert(
      tile_size_m == matA_t::tile_size_y,
      "matAcc tile m should match with matA tile m");
  static_assert(
      a_tile_size_x == b_tile_size_y,
      "matA tile k should match with matB tile k");
  static_assert(
      tile_size_n == matB_t::tile_size_x,
      "matAcc tile n should match with matB tile n");
  static_assert(
      block_size_m == a_block_size_y,
      "matAcc block m should match with matA block m");
  static_assert(
      block_size_n == b_block_size_x,
      "matAcc block n should match with matB block n");
  static_assert(
      (tile_size_k % block_size_k) == 0,
      "matAcc tile_size_k should be a multiple of block_size_k");

  static constexpr int32_t num_block_n = matDst_t::num_block_x;
  static constexpr int32_t num_block_k = tile_size_k / block_size_k;

  static constexpr auto b_reg_sizes = b_block_size_y * b_tile_size_x;
  static constexpr auto tile_n_elems = num_block_n * block_elems;
  static constexpr auto a_tile_k_elems = num_block_k * a_block_elems;
  static constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
  static constexpr uint32_t a_tail_blk_elems = a_block_size_x * a_tail_blk_w;
  static constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;

  using mma_attr = mma_attr_t<arch_tag_, mma_engine::fpu, tile_size_m>;
  static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;

  using blk_mma = blk_mma_t<
      dtype_dst,
      dtype_src,
      dtype_src,
      dtype_a,
      block_size_m,
      block_size_n,
      block_size_k,
      mma_m,
      matA_t::register_layout,
      matB_t::register_layout,
      mma_engine::fpu>;

  __XETLA_API static void mma(
      matDst_t& dst,
      matSrc_t& src,
      matB_t& b,
      matA_t& a) {
    { // k_blk=0
      auto b_reg = xetla_cvt<dtype_src, dtype_b, b_reg_sizes>(
          b.reg.xetla_select<b_reg_sizes, 1>(0));
      if constexpr (blk_m_iters >= 1) {
#pragma unroll
        for (uint32_t i = 0; i < blk_m_iters; i++) {
          auto a_block =
              a.reg.xetla_select<a_block_elems, 1>(i * a_tile_k_elems);
          auto i_start_off = i * tile_n_elems;
#pragma unroll
          for (uint32_t j = 0; j < num_block_n; j++) {
            auto b_block =
                b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
            auto src_block = src.reg.xetla_select<block_elems, 1>(
                i_start_off + j * block_elems);
            auto dst_block = dst.reg.xetla_select<block_elems, 1>(
                i_start_off + j * block_elems);
            blk_mma::mma_core(dst_block, src_block, b_block, a_block);
          }
        }
      }

      // process the tail
      if constexpr (tail_size_m != 0) {
        using tail_blk_mma = blk_mma_t<
            dtype_dst,
            dtype_src,
            dtype_src,
            dtype_a,
            tail_size_m,
            block_size_n,
            block_size_k,
            mma_m,
            matA_t::register_layout,
            matB_t::register_layout,
            mma_engine::fpu>;

        auto a_block =
            a.reg.xetla_select<a_tail_blk_elems, 1>(a_tail_start_offset);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto src_block = src.reg.xetla_select<acc_tail_blk_elems, 1>(
              tail_start_offset + j * acc_tail_blk_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              tail_start_offset + j * acc_tail_blk_elems);
          tail_blk_mma::mma_core(dst_block, src_block, b_block, a_block);
        }
      }
    }
    // different K block
#pragma unroll
    for (uint32_t k_i = 1; k_i < num_block_k; k_i++) {
      xetla_vector<dtype_src, b_reg_sizes> b_reg =
          xetla_cvt<dtype_src, dtype_b, b_reg_sizes>(
              b.reg.xetla_select<b_reg_sizes, 1>(
                  k_i * b_block_size_y * b_tile_size_x));

      if constexpr (blk_m_iters >= 1) {
#pragma unroll
        for (uint32_t i = 0; i < blk_m_iters; i++) {
          auto a_block = a.reg.xetla_select<a_block_elems, 1>(
              i * a_tile_k_elems + k_i * a_block_elems);
          auto i_start_off = i * tile_n_elems;
#pragma unroll
          for (uint32_t j = 0; j < num_block_n; j++) {
            auto b_block =
                b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
            auto dst_block = dst.reg.xetla_select<block_elems, 1>(
                i_start_off + j * block_elems);
            blk_mma::mma_core(dst_block, dst_block, b_block, a_block);
          }
        }
      }

      // process the tail
      if constexpr (tail_size_m != 0) {
        using tail_blk_mma = blk_mma_t<
            dtype_dst,
            dtype_src,
            dtype_src,
            dtype_a,
            tail_size_m,
            block_size_n,
            block_size_k,
            mma_m,
            matA_t::register_layout,
            matB_t::register_layout,
            mma_engine::fpu>;

        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tail_start_offset + k_i * a_tail_blk_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              tail_start_offset + j * acc_tail_blk_elems);
          tail_blk_mma::mma_core(dst_block, dst_block, b_block, a_block);
        }
      }
    }
  }
};

} // namespace gpu::xetla::subgroup
