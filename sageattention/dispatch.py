"""
Dispatch logic for attention backends
"""
import os
from functools import lru_cache
from .specs import AttentionSpec

_DEFAULT_SM75_MIN_SAGE_LEN = 1024


def _read_sm75_min_sage_len():
    try:
        return int(os.environ.get("SAGEATTN_SM75_MIN_LEN", _DEFAULT_SM75_MIN_SAGE_LEN))
    except ValueError:
        return _DEFAULT_SM75_MIN_SAGE_LEN


# On sm75 the INT8 path loses to torch SDPA when either sequence is shorter than this
SM75_MIN_SAGE_LEN = _read_sm75_min_sage_len()


@lru_cache(maxsize=256)
def select_backend(spec: AttentionSpec):
    """
    纯函数：给定 spec（不含实际 tensor 数据，只含形状/dtype/flag），返回一个 backend 名字或 callable。
    lru_cache 生效的前提是 AttentionSpec 是 frozen dataclass（可 hash），
    绝不能把 tensor 本身放进 spec 里，否则 lru_cache 会持有 tensor 引用导致显存泄漏。
    """
    if spec.arch != "sm75":
        return "torch_sdpa_or_other_backend"

    # torch SDPA cannot return LSE, so this case always stays on the INT8 path
    if spec.return_lse:
        return "sm75_qattn_long"

    if min(spec.q_len, spec.kv_len) < SM75_MIN_SAGE_LEN:
        return "torch_sdpa"

    return "sm75_qattn_long"
