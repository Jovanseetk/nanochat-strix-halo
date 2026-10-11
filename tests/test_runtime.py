"""Hardware selection tests run without an AMD or NVIDIA GPU."""

from types import SimpleNamespace
import sys

import pytest
import torch

from nanochat import common
from nanochat import flash_attention


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.delenv("NANOCHAT_DTYPE", raising=False)
    monkeypatch.delenv("NANOCHAT_COMPILE", raising=False)
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(
        gcnArchName="gfx1151:sramecc-:xnack-"))
    # HIP's compatibility value must never be interpreted as an NVIDIA SM.
    def no_nvidia_capability(*args):
        raise AssertionError("queried NVIDIA capability on ROCm")
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_nvidia_capability)


@pytest.mark.parametrize("bf16_supported, expected", [(True, torch.bfloat16), (False, torch.float32)])
def test_rocm_dtype_uses_bf16_support(rocm, monkeypatch, bf16_supported, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)
    dtype, reason = common._detect_compute_dtype()
    assert dtype == expected
    assert "ROCm" in reason


def test_explicit_dtype_takes_precedence(rocm, monkeypatch):
    monkeypatch.setenv("NANOCHAT_DTYPE", "float32")
    assert common._detect_compute_dtype()[0] == torch.float32


@pytest.mark.parametrize("capability, expected", [((9, 0), torch.bfloat16), ((7, 5), torch.float32)])
def test_nvidia_dtype_unchanged(monkeypatch, capability, expected):
    monkeypatch.delenv("NANOCHAT_DTYPE", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    assert common._detect_compute_dtype()[0] == expected


def test_rocm_does_not_load_nvidia_flash_attention(rocm, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(
        get_kernel=lambda *a: SimpleNamespace(flash_attn_interface=object()),
        has_kernel=lambda *a: True,
    ))
    assert flash_attention._load_flash_attention_3() is None


def test_strix_halo_defaults_to_eager(rocm):
    assert not common.compile_enabled()


def test_other_rocm_gpus_keep_compilation(rocm, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(gcnArchName="gfx942"))
    assert common.compile_enabled()


@pytest.mark.parametrize("override, expected", [("0", False), ("1", True)])
def test_compile_override(rocm, monkeypatch, override, expected):
    monkeypatch.setenv("NANOCHAT_COMPILE", override)
    assert common.compile_enabled() is expected


def test_invalid_compile_override_is_actionable(monkeypatch):
    monkeypatch.setenv("NANOCHAT_COMPILE", "offf")
    with pytest.raises(ValueError, match="NANOCHAT_COMPILE"):
        common.compile_enabled()


def test_eager_wrapper_preserves_module_and_gradients(monkeypatch):
    monkeypatch.setenv("NANOCHAT_COMPILE", "0")
    model = torch.nn.Linear(3, 2)
    wrapped = common.maybe_compile(model, dynamic=False)
    assert wrapped is model
    wrapped(torch.ones(1, 3)).sum().backward()
    torch.testing.assert_close(model.weight.grad, torch.ones(2, 3))


def test_eager_decorator_runs_without_compiler(monkeypatch):
    monkeypatch.setenv("NANOCHAT_COMPILE", "0")
    def forbidden(*args, **kwargs):
        raise AssertionError("eager mode invoked torch.compile")
    monkeypatch.setattr(torch, "compile", forbidden)

    @common.maybe_compile(fullgraph=True)
    def square(x):
        return x.square()

    torch.testing.assert_close(square(torch.tensor([3.])), torch.tensor([9.]))


def test_strix_halo_rejects_fp8(rocm):
    assert not common.supports_fp8()


@pytest.mark.parametrize("capability, supported", [((8, 0), False), ((9, 0), True)])
def test_fp8_requires_supported_nvidia_gpu(monkeypatch, capability, supported):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    assert common.supports_fp8() is supported
