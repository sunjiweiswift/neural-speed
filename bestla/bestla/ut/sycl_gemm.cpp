#include "bestla_ut.h"
#include "sycl_ut.h"
#include "../sycl/sycl_wrapper.h"
#include "bestla_prologue_b.h"
namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace sycl_ut {

class UT_SyclInt4Dequant {
 public:
  UT_SyclInt4Dequant() {
    UT_START();
    ut_fp32_T(1024, 1024, 32);
    ut_fp16_x8_T(1024, 1024, 32);
    ut_bf16_x8_T(1024, 1024, 32);
  }

  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        ref[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS4Trans<sycl_gemm::xve::DefaultSGemmCore, float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::dequant_s4<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);

    avector<float> refNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(ref.data(), refNT.data(), n, k, k, n);
    ev = ProB::dequant_s4_trans<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(refNT.data(), dequant.data(), dequant.size(), 0.001f);
  }


  void ut_bf16_x8_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<int32_t> rawB(k * n / 8);
    int blks = updiv(k, blocksize);
    avector<utils::bf16> scale(blks * n), zps(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), (utils::bf16)0.01f, (utils::bf16)0.03f);
    fill_buffer_randn(zps.data(), zps.size(), (utils::bf16)0.01f, (utils::bf16)0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), int32_t(0), int32_t(255));
    auto srcptr = rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 8) {
        auto tmp = srcptr[i * k / 8 + j / 8];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = static_cast<float>(((tmp      ) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 1] = static_cast<float>(((tmp >> 4 ) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 2] = static_cast<float>(((tmp >> 8 ) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 3] = static_cast<float>(((tmp >> 12) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 4] = static_cast<float>(((tmp >> 16) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 5] = static_cast<float>(((tmp >> 20) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 6] = static_cast<float>(((tmp >> 24) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 7] = static_cast<float>(((tmp >> 28) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS4x8Trans<sycl_gemm::xve::DefaultBGemmCore, sycl::ext::oneapi::bfloat16>;
    sycl_vector<sycl::ext::oneapi::bfloat16> dS(scale.size(), q), dZ(zps.size(), q), dequantB(n * k, q);
    sycl_vector<int32_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dZ.data(), zps.data(), zps.size() * 2).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 4).wait();
    auto S_d = dS.data();
    auto Z_d = dZ.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::dequant_s4x8<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks, Z_d}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * sizeof(utils::bf16)).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), (utils::bf16)0.001f);
  }

    void ut_fp16_x8_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<int32_t> rawB(k * n / 8);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), zps(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), (utils::fp16)0.01f, (utils::fp16)0.03f);
    fill_buffer_randn(zps.data(), zps.size(), (utils::fp16)0.01f, (utils::fp16)0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), int32_t(0), int32_t(255));
    auto srcptr = rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 8) {
        auto tmp = srcptr[i * k / 8 + j / 8];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = static_cast<float>(((tmp      ) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 1] = static_cast<float>(((tmp >> 4 ) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 2] = static_cast<float>(((tmp >> 8 ) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 3] = static_cast<float>(((tmp >> 12) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 4] = static_cast<float>(((tmp >> 16) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 5] = static_cast<float>(((tmp >> 20) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 6] = static_cast<float>(((tmp >> 24) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
        ref[i * k + j + 7] = static_cast<float>(((tmp >> 28) & 0x0f) - 8) * (float)scale[noffset] + (float)zps[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS4x8Trans<sycl_gemm::xve::DefaultHGemmCore, sycl::half>;
    sycl_vector<sycl::half> dS(scale.size(), q), dZ(zps.size(), q), dequantB(n * k, q);
    sycl_vector<int32_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dZ.data(), zps.data(), zps.size() * 2).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 4).wait();
    auto S_d = dS.data();
    auto Z_d = dZ.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::dequant_s4x8<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks, Z_d}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * sizeof(utils::fp16)).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), (utils::fp16)0.001f);
  }
};
static UT_SyclInt4Dequant sUT_SyclInt4Dequant;

class UT_SyclS4Gemv {
 public:
  UT_SyclS4Gemv() {
    UT_START();
    ut_T(1024, 11008, 32);
    ut_T(1024, 1024, 32);
    ut_half(1024, 11008, 32);
    ut_half(1024, 1024, 32);
    ut_x8_half(1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemm_t>;

  void ut_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(A.data(), A.size(), -0.1f, 0.3f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 2;
    int constexpr GroupK = SgSize * TileK;
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = ProBTransT<SGemm_t>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k, blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_x8_half(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), zp(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(zp.data(), zp.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(A.data(), A.size(), utils::fp16(-0.1f), utils::fp16(0.3f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        float fscale = float(scale[noffset]);
        float fzp = float(zp[noffset]);
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * fscale + fzp;
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * fscale + fzp;
      }
    }
    gemmref_fp16fp16fp16(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<sycl::half> dS(scale.size(), q), dZ(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dZ.data(), zp.data(), zp.size() * 2).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 2).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    auto S_d = dS.data();
    auto Z_d = dZ.data();
    auto A_d = dA.data();
    auto B_d = reinterpret_cast<int32_t*>(dB.data());
    auto C_d = dC.data();
    auto ev = sycl_prologue_b::WeightS4x8Trans<xve::DefaultHGemmCore, sycl::half>::gemv(A_d, {B_d, S_d, blks, Z_d}, C_d, n, k,
                                                                                      blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 2).wait();
    buffer_error(refC.data(), C.data(), C.size(), utils::fp16(0.1f));
  }

  void ut_half(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(A.data(), A.size(), utils::fp16(-0.1f), utils::fp16(0.3f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        float fscale = float(scale[noffset]);
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * fscale;
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * fscale;
      }
    }
    gemmref_fp16fp16fp16(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<sycl::half> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 2).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = sycl_prologue_b::WeightS4Trans<xve::DefaultHGemmCore, sycl::half>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k,
                                                                                      blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 2).wait();
    buffer_error(refC.data(), C.data(), C.size(), utils::fp16(0.1f));
  }
};
static UT_SyclS4Gemv sUT_SyclS4Gemv;

void mha_sref(float* Q, float* K, float* V, float* S, float* O, int batch, int seq, int seqA, int hnum, int hsize) {
  avector<float> tmps(seqA);
  int nf = hnum * hsize;
  const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
  int n_past = seqA - seq;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < seq; j++) {
      for (int ii = 0; ii < hnum; ii++) {
        float maxs = 0.f;
        for (int jj = 0; jj < seqA; jj++) {
          float tmp = 0.f;
          if (jj <= j + n_past) {
            for (int kk = 0; kk < hsize; kk++) {
              tmp +=
                  Q[i * seq * nf + j * nf + ii * hsize + kk] * K[i * nf * seqA + ii * seqA * hsize + jj * hsize + kk];
            }
            tmp *= attn_scale;
          } else {
            tmp = -INFINITY;
          }

          tmps[jj] = tmp;
          maxs = std::max(maxs, tmp);
        }
        float sums = 0.f;
        for (int jj = 0; jj < seqA; jj++) {
          tmps[jj] = expf(tmps[jj] - maxs);
          sums += tmps[jj];
        }
        sums = 1.f / sums;
        for (int jj = 0; jj < seqA; jj++) {
          tmps[jj] *= sums;
          S[i * seq * hnum * seqA + j * hnum * seqA + ii * seqA + jj] = tmps[jj];
        }
        for (int kk = 0; kk < hsize; kk++) {
          float tmp = 0.f;
          for (int jj = 0; jj < seqA; jj++) {
            tmp += tmps[jj] * V[i * nf * seqA + ii * hsize * seqA + kk * seqA + jj];
          }
          O[i * seq * nf + j * nf + ii * hsize + kk] = tmp;
        }
      }
    }
  }
}

class UT_MHASgemm {
 public:
  UT_MHASgemm() {
    UT_START();
    ut_T(1, 1, 1, 32, 128);
    ut_T(1, 1, 64, 32, 128);
    ut_T(4, 1, 64, 32, 128);
    ut_T(4, 64, 64, 32, 128);
  }
  template <typename T, typename T_DST>
  class MHA {
   public:
    template <bool Mask>
    static sycl::event forward(int batch, int seq, int seq_acc, int hnum, int hsize, const T* Q, const T* K, const T* V,
                               T_DST* O, sycl::queue* q) {
      const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
      int constexpr SgSize = 16;
      assert(hsize % SgSize == 0);
      int n_past = seq_acc - seq;
      if constexpr (Mask) {
        assert(seq > 1);
      }
      int WgSize = SgSize;
      int seq_acc_pad = utils::padto_le(seq_acc, WgSize * 2);
      int nf = hnum * hsize;
      auto ev = q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> slm(sycl::range(std::max(seq_acc, 1024)), cgh);
        cgh.parallel_for(sycl::nd_range<1>(WgSize * batch * seq * hnum, WgSize),
                         [=](auto it) [[intel::reqd_sub_group_size(SgSize)]] {
                           auto sg = it.get_sub_group();
                           auto sg_idx = sg.get_group_id()[0];
                           auto wg_idx = it.get_group(0);
                           auto wg_loc_id = it.get_local_id();
                           auto lane_id = sg.get_local_id()[0];

                           int i = wg_idx;
                           int ih = i % hnum;
                           i /= hnum;
                           int is = i % seq;
                           i /= seq;
                           int ib = i % batch;
                           size_t Q_off = ib * seq * nf + is * nf + ih * hsize;
                           size_t K_off = ib * seq_acc * nf + ih * hsize * seq_acc;
                           size_t V_off = ib * seq_acc * nf + ih * hsize * seq_acc;
                           size_t O_off = ib * seq * nf + is * nf + ih * hsize;
                           typedef sycl::vec<T, 2> TC;
                           T maxs = -INFINITY;
                           for (int jj = 0; jj < seq_acc; jj++) {
                             TC tmp = {0, 0};
                             if constexpr (Mask) {
                               if (jj <= is + n_past) {
                                 for (int ik = wg_loc_id * 2; ik < hsize; ik += WgSize * 2) {
                                   tmp += *(TC*)&Q[Q_off + ik] * *(TC*)&K[K_off + jj * hsize + ik];
                                 }
                                 tmp *= attn_scale;
                               } else {
                                 tmp = {-INFINITY, -INFINITY};
                               }
                             } else {
                               for (int ik = wg_loc_id * 2; ik < hsize; ik += WgSize * 2) {
                                 tmp += *(TC*)&Q[Q_off + ik] * *(TC*)&K[K_off + jj * hsize + ik];
                               }
                               tmp *= attn_scale;
                             }
                             T tmp_sum = tmp[0] + tmp[1];
                             T sum = 0;
                             for (int i = 0; i < SgSize; i += 1) {
                               sum += sg.shuffle(tmp_sum, i);
                             }
                             slm[jj] = sum;
                             maxs = std::max(maxs, sum);
                           }
                           float fsums = 0.f;
                           float fmax = float(maxs);
                           int jj = wg_loc_id * 2;
                           for (; jj < seq_acc_pad; jj += WgSize * 2) {
                             auto s2 = *(TC*)&slm[jj];
                             s2[0] = expf(s2[0] - fmax);
                             s2[1] = expf(s2[1] - fmax);
                             fsums += s2[0];
                             fsums += s2[1];
                             *(TC*)&slm[jj] = s2;
                           }
                           if (jj < seq_acc) {
                             slm[jj] = expf(float(slm[jj]) - fmax);
                             fsums += slm[jj];
                             if (jj + 1 < seq_acc) {
                               slm[jj + 1] = expf(float(slm[jj + 1]) - fmax);
                               fsums += slm[jj + 1];
                             }
                           }
                           float gsum = 0;
                           for (int i = 0; i < SgSize; i += 1) {
                             gsum += sg.shuffle(fsums, i);
                           }
                           T scale = 1.f / gsum;
                           jj = wg_loc_id * 2;
                           for (; jj < seq_acc_pad; jj += WgSize * 2) {
                             auto s2 = *(TC*)&slm[jj];
                             s2 *= scale;
                             *(TC*)&slm[jj] = s2;
                           }
                           if (jj < seq_acc) {
                             slm[jj] *= scale;
                             if (jj + 1 < seq_acc) {
                               slm[jj + 1] *= scale;
                             }
                           }

                           for (int kk = 0; kk < hsize; kk++) {
                             TC tmp = {0, 0};
                             jj = wg_loc_id * 2;
                             for (; jj < seq_acc_pad; jj += WgSize * 2) {
                               auto s2 = *(TC*)&slm[jj];
                               auto v2 = *(TC*)&V[V_off + kk * seq_acc + jj];
                               tmp += s2 * v2;
                             }
                             if (jj < seq_acc) {
                               tmp[0] += slm[jj] * V[V_off + kk * seq_acc + jj];
                               if (jj + 1 < seq_acc) {
                                 tmp[1] += slm[jj + 1] * V[V_off + kk * seq_acc + jj + 1];
                               }
                             }
                             T tmp_sum = tmp[0] + tmp[1];
                             T sum = 0;
                             for (int i = 0; i < SgSize; i += 1) {
                               sum += sg.shuffle(tmp_sum, i);
                             }
                             O[O_off + kk] = sum;
                           }
                         });
      });
      return ev;
    }
  };

  void ut_T(int batch, int seq, int seqA, int hnum, int hsize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    assert(seqA >= seq);
    printf("Test Case %s: %d %d %d %d %d Device:%s\n", __FUNCTION__, batch, seq, seqA, hnum, hsize,
           dev->getName().c_str());
    avector<float> Q(batch * seq * hnum * hsize), K(batch * seqA * hnum * hsize), V(batch * seqA * hnum * hsize);
    fill_buffer_randn(Q.data(), Q.size(), -0.5f, 0.5f);
    fill_buffer_randn(K.data(), K.size(), -0.5f, 0.5f);
    fill_buffer_randn(V.data(), V.size(), -0.5f, 0.5f);
    avector<float> S(batch * seq * hnum * seqA), O(batch * seq * hnum * hsize);
    mha_sref(Q.data(), K.data(), V.data(), S.data(), O.data(), batch, seq, seqA, hnum, hsize);
    sycl_vector<float> dQ(batch * seq * hnum * hsize, q), dK(batch * seqA * hnum * hsize, q),
        dV(batch * seqA * hnum * hsize, q);
    sycl_vector<float> dS(batch * seq * hnum * seqA, q), dO(batch * seq * hnum * hsize, q);
    q->memcpy(dQ.data(), Q.data(), Q.size() * sizeof(Q[0]));
    q->memcpy(dK.data(), K.data(), K.size() * sizeof(K[0]));
    q->memcpy(dV.data(), V.data(), V.size() * sizeof(V[0]));
    q->wait();
    auto Qptr = dQ.data();
    auto Kptr = dK.data();
    auto Vptr = dV.data();
    auto Sptr = dS.data();
    auto Optr = dO.data();
    int nf = hnum * hsize;
    int n_past = seqA - seq;
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
    if (seq > 1) {
      MHA<float, float>::forward<true>(batch, seq, seqA, hnum, hsize, Qptr, Kptr, Vptr, Optr, q).wait();
    } else {
      MHA<float, float>::forward<false>(batch, seq, seqA, hnum, hsize, Qptr, Kptr, Vptr, Optr, q).wait();
    }
    // auto ev = q->submit([&](sycl::handler& cgh) {
    //   cgh.parallel_for(num_items, [=](auto it) {
    //     int i = it;
    //     int ih = i % hnum;
    //     i /= hnum;
    //     int is = i % seq;
    //     i /= seq;
    //     int ib = i % batch;
    //     float maxs = 0.f;
    //     float tmps[64];
    //     for (int jj = 0; jj < seqA; jj++) {
    //       float tmp = 0.f;
    //       if (jj <= is + n_past) {
    //         for (int kk = 0; kk < hsize; kk++) {
    //           tmp += Qptr[ib * seq * nf + is * nf + ih * hsize + kk] *
    //                  Kptr[ib * nf * seqA + kk + ih * seqA * hsize + jj * hsize];
    //         }
    //         tmp *= attn_scale;
    //       } else {
    //         tmp = -INFINITY;
    //       }

    //      tmps[jj] = tmp;
    //      maxs = std::max(maxs, tmp);
    //    }
    //    float sums = 0.f;
    //    for (int jj = 0; jj < seqA; jj++) {
    //      tmps[jj] = expf(tmps[jj] - maxs);
    //      sums += tmps[jj];
    //    }
    //    sums = 1.f / sums;
    //    for (int jj = 0; jj < seqA; jj++) {
    //      tmps[jj] *= sums;
    //      Sptr[ib * seq * hnum * seqA + is * hnum * seqA + ih * seqA + jj] = tmps[jj];
    //    }
    //    for (int kk = 0; kk < hsize; kk++) {
    //      float tmp = 0.f;
    //      for (int jj = 0; jj < seqA; jj++) {
    //        tmp += tmps[jj] * Vptr[ib * seqA * nf + jj + ih * hsize * seqA + kk * seqA];
    //      }
    //      Optr[ib * seq * nf + is * nf + ih * hsize + kk] = tmp;
    //    }
    //  });
    //});
    q->wait();
    avector<float> STar(batch * seq * hnum * seqA), OTar(batch * seq * hnum * hsize);
    q->memcpy(STar.data(), Sptr, STar.size() * sizeof(STar[0]));
    q->memcpy(OTar.data(), Optr, OTar.size() * sizeof(OTar[0]));
    q->wait();
    // buffer_error(S.data(), STar.data(), S.size(), 0.001f);
    buffer_error(O.data(), OTar.data(), O.size(), 0.001f);
  }
};
static UT_MHASgemm sUT_MHASgemm;
}  // namespace sycl_ut
}  // namespace bestla
