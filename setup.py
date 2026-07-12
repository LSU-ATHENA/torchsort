#!/usr/bin/env python3

import os
import shutil
import sys
from functools import lru_cache
from subprocess import DEVNULL, call

import torch
from setuptools import setup
from torch.utils import cpp_extension


@lru_cache(None)
def cuda_toolkit_available():
    """Return whether a CUDA toolkit with nvcc is available for extension builds."""
    nvcc = shutil.which("nvcc")
    if nvcc is None and cpp_extension.CUDA_HOME:
        candidate = os.path.join(cpp_extension.CUDA_HOME, "bin", "nvcc")
        if os.path.isfile(candidate):
            nvcc = candidate

    if nvcc is None:
        if os.getenv("TORCHSORT_FORCE_CUDA") == "1" or os.getenv("FORCE_CUDA") == "1":
            raise RuntimeError(
                "CUDA build was forced, but nvcc was not found. "
                "Install a CUDA toolkit with nvcc and set CUDA_HOME accordingly."
            )
        # torch.cuda.is_available() only indicates runtime availability; compiling
        # custom CUDA extensions still requires nvcc.
        if torch.cuda.is_available():
            print(
                "[torchsort] CUDA runtime is available but nvcc was not found; "
                "building CPU-only extension."
            )
        return False

    # `nvcc` without arguments may return a non-zero exit code on some CUDA
    # toolkits even when installed correctly. Probe with `--version` instead.
    return call([nvcc, "--version"], stdout=DEVNULL, stderr=DEVNULL) == 0


def compile_args():
    args = ["-fopenmp", "-ffast-math"]
    if sys.platform == "darwin":
        return ["-Xpreprocessor"] + args
    return args


def ext_modules():
    extensions = [
        cpp_extension.CppExtension(
            "torchsort.isotonic_cpu",
            sources=["torchsort/isotonic_cpu.cpp"],
            extra_compile_args=compile_args(),
        ),
    ]
    if cuda_toolkit_available():
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        extensions.append(
            cpp_extension.CUDAExtension(
                "torchsort.isotonic_cuda",
                sources=["torchsort/isotonic_cuda.cu"],
                extra_compile_args={
                    "cxx": compile_args(),
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            ),
        )
    return extensions

setup(
    name="torchsort",
    version="0.1.10" + os.getenv("TORCHSORT_VERSION_SUFFIX", ""),
    packages=["torchsort"],
    ext_modules=ext_modules(),
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    include_package_data=True,
)
