# SageAttention-Turing

SageAttention2 adapted for NVIDIA Turing GPUs (SM75: Tesla T4, RTX 20xx), aimed at running quantized attention in ComfyUI on a Colab T4.

Upstream project: https://github.com/thu-ml/SageAttention (it does not support Turing; this fork ports the Ampere kernels to SM75).

Credits: the MMA fix for Turing is based on https://github.com/mit-han-lab/nunchaku.

## Status

Tested configuration:

| Component | Version |
|---|---|
| GPU | Tesla T4 (compute capability 7.5) |
| PyTorch | 2.15.0+cu132 |
| CUDA toolkit (nvcc) | 13.2 |
| Triton | 3.9.0 |
| Python | 3.13 |

The extensions `_qattn_sm80`, `_fused`, `_sm75_fast_dispatch` and `_sm75_short_sdpa_v2` build and import in this configuration. No NaNs were observed in the benchmark cases below.

Not evaluated: end-to-end image or video quality in ComfyUI. Accuracy was only compared against `torch.nn.functional.scaled_dot_product_attention` on synthetic random inputs.

## How attention is routed on SM75

`sageattn()` selects a backend in `sageattention/dispatch.py`:

| Condition | Backend |
|---|---|
| `return_lse=True` | INT8 path (`sm75_qattn_long`) |
| `min(q_len, kv_len) < 1024` | `torch.nn.functional.scaled_dot_product_attention` |
| otherwise | INT8 path (`sm75_qattn_long`): per-warp INT8 QK, FP16 PV with FP32 accumulation |

The threshold can be changed without rebuilding:

```bash
export SAGEATTN_SM75_MIN_LEN=1024   # default
```

An invalid value falls back to 1024.

## Changes in this fork

1. **SDPA routing for short and cross-attention sequences** (`dispatch.py`, `core.py`).
   Before, any call with `max(q_len, kv_len) < 1024` went through the custom short kernels, and cross-attention with a short `kv` (for example 77 text tokens) went through the INT8 path. On a T4 both were slower than torch SDPA (0.12x to 0.53x for short self-attention, 0.32x for cross-attention). Such calls now use torch SDPA directly through the new `torch_sdpa` backend, without the v2 -> v1 -> SDPA fallback chain.
2. **Default softmax scale** (`backends/sm75_short.py`).
   When `sm_scale=None`, the custom short kernels received `0.0`, which produced wrong results for short sequences (`max_abs_diff` of about 1.2 against SDPA at 64 tokens). `resolve_sm_scale()` now substitutes `head_dim ** -0.5`.
3. **Shared SDPA fallback** (`backends/sm75_short.py`).
   The final PyTorch fallback was extracted into `torch_sdpa_fallback()` and is used by both the short backend and the new `torch_sdpa` route. Existing function names are unchanged.
4. **Build compatibility with PyTorch 2.15**: see "Build" below. The C++ standard must be raised to C++20.

The `sm75_short` branch is still present in `core.py`, but `select_backend()` no longer returns it.

## Benchmarks (Tesla T4)

Setup: fp16, batch size 1, 10 heads (8 for head_dim 128), HND layout built with `transpose(1, 2)` (last dimension contiguous), random normal inputs, CUDA events, 20 timed iterations after 5 warmup iterations. Baseline is `scaled_dot_product_attention`.

These numbers were measured **before** change 1. Rows marked "now SDPA" are routed to torch SDPA in the current code and have not been re-measured; their expected speedup is about 1.0x.

| Shape | SDPA, ms | Sage, ms | Speedup | max abs diff | Current route |
|---|---|---|---|---|---|
| self, N=64, D=64 | 0.059 | 0.483 | 0.12x | 1.2062 | now SDPA |
| self, N=256, D=64 | 0.060 | 0.113 | 0.53x | 0.0000 | now SDPA |
| self, N=1024, D=64 | 0.502 | 0.346 | 1.45x | 0.0122 | INT8 |
| self, N=4096, D=64 | 5.402 | 1.765 | 3.06x | 0.0049 | INT8 |
| cross, Q=4096, KV=77, D=64 | 0.152 | 0.467 | 0.32x | 0.0264 | now SDPA |
| self, N=4096, D=128 | 5.603 | 2.671 | 2.10x | 0.0025 | INT8 |
| self, N=16384, D=64 | 52.866 | 26.449 | 2.00x | 0.0018 | INT8 |

## Known limitations

- `attn_mask` is not supported. `sageattn()` accepts it through `**kwargs` and ignores it.
- Only head_dim 64 and 128 were benchmarked.
- Sequences shorter than `SAGEATTN_SM75_MIN_LEN` and cross-attention with a short key/value length are not accelerated; they run on torch SDPA.
- Speedups depend on how much of a model's runtime is spent in long self-attention.

## Build

Requirements: an `nvcc` that matches the CUDA version of the installed PyTorch build (CUDA 13.2 for `cu132`), and the target GPU visible during the build. `setup.py` selects the architecture from the visible GPUs, falls back to 7.5 when none is visible, and does not read `TORCH_CUDA_ARCH_LIST`.

```bash
git clone https://github.com/<your-username>/SageAttention-Turing.git
cd SageAttention-Turing
# Required for PyTorch >= 2.15: its headers need C++20
sed -i 's/-std=c++17/-std=c++20/g' setup.py
python setup.py bdist_wheel --dist-dir=wheels
```

### PyTorch <= 2.10

If dependency constraints force you to use PyTorch 2.10 or older, edit `python\Lib\site-packages\torch\include\torch\csrc\dynamo\compiled_autograd.h` manually.

Original code:

```cpp
    } else if constexpr (::std::is_same_v<T, ::std::string>) {
      return at::StringType::get();
    }
```

Modified code:

```cpp
    }/* else if constexpr (::std::is_same_v<T, ::std::string>) {
      return at::StringType::get();
    } */
```

## History (original notes)

Status as of 2.1.1:

- Compiles on CUDA 11.8.
- Fused kernel: working on SM75.
- qattn: compiled and ran when selected, but produced NaNs. Output quality was low; it was only tested on SDXL.
- Sparge attention could, in theory, run given the same treatment.

9/1/25: Triton with the fused kernel works in ComfyUI with the SageAttention command-line option.
