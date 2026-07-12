# [Torchsort](https://github.com/teddykoker/torchsort)
## [LSU ATHENA Fork for Blackwell Silicon](https://github.com/LSU-ATHENA)

This is a fork of the existing torchsort repository with some minor changes to make it compile and run with newer Python (3.12) and PyTorch (2.13) versions as well as work on NVIDIA Blackwell Silicon, such as a 50-series GPU, RTX Pro 5000/6000 or DGX Spark. Beyond this paragraph, the only part of this README that has been changed is the installation instructions. 

---

Fast, differentiable sorting and ranking in PyTorch.

Pure PyTorch implementation of [Fast Differentiable Sorting and
Ranking](https://arxiv.org/abs/2002.08871) (Blondel et al.). Much of the code is
copied from the original Numpy implementation at
[google-research/fast-soft-sort](https://github.com/google-research/fast-soft-sort),
with the isotonic regression solver rewritten as a PyTorch C++ and CUDA
extension.

## Install
First, you will want a Python ~3.12 environment with [PyTorch ~2.13](https://pytorch.org/get-started/locally/) installed:
```
conda create -n torchsort python=3.12 -y
conda activate torchsort
pip install ipython
pip install torch torchvision
```

We first install ipython to probe if PyTorch can access the GPU:
```
ipython
import torch as t
t.cuda.is_available()
```
If this returns back `True`, we're good; if `False`, you need to toy around with torch/cuda versions until it works for your GPU. The difficulty of this will depend on the specifications of your system - OS, Hardware, Driver, etc.

Second, you will need to install `nvcc`. This is NVIDIA's compiler for CUDA. It it crucial for compiling and running torchsort.
```
conda install -y -c nvidia cuda-nvcc=13.0
```
**Note**: My system is running CUDA 13.0 per `nvidia-smi` which is why I specified it as much. You will likely be running 12.8 - 13.2 give or take, especially for Blackwell. If running Blackwell, set this environment variable:
```
export TORCH_CUDA_ARCH_LIST="12.0"
```

You will then identify where nvcc is installed on your environment:
```
which nvcc
```
On my system, this outputs `/home/kgmills/miniconda3/autobuild/bin/nvcc`. Given this, we now want to set an environment variable to *tell* torchsort where to find `nvcc`.
```
export CUDA_HOME='/home/kgmills/miniconda3/envs/autobuild/'
```
**Note**: I removed `bin/nvcc` from the export. 


Third, you will want to clone this repo **as a standalone directory that is not nestled with another repo**. E.g., if you are cloning this to get `torchsort` for [`AutoBuild`](https://github.com/Ascend-Research/AutoBuild), you should **not** nestle this repo within a cloned `AutoBuild` repo. 

```
git clone https://github.com/LSU-ATHENA/torchsort.git
cd torchsort
pip install -v -e . --no-build-isolation --no-deps
```

This should be all that's needed - if you encounter other issues, no-a-days we have Codex and Claude to decipher such issues and help. 

If the package compiles and installs successfully, you can additionally confirm using ipython:

```
import torch as t
import torchsort
x = t.tensor([[3., 1., 2.]], requires_grad=True)
if t.cuda.is_available():
  x = x.cuda()
else:
  print("Torch does not have access to CUDA!")

y = torchsort.soft_rank(x, regularization_strength=1.0)
(y.sum()).backward()
print("soft_rank ok:", y)
print("grad ok:", x.grad)
```

This may throw some warnings at you but ultimately what is important is that it does not fail. If its good, we've installed `torchsort` for Blackwell!

## Usage

`torchsort` exposes two functions: `soft_rank` and `soft_sort`, each with
parameters `regularization` (`"l2"` or `"kl"`) and `regularization_strength` (a
scalar value). Each will rank/sort the last dimension of a 2-d tensor, with an
accuracy dependent upon the regularization strength:

```python
import torch
import torchsort

x = torch.tensor([[8, 0, 5, 3, 2, 1, 6, 7, 9]])

torchsort.soft_sort(x, regularization_strength=1.0)
# tensor([[0.5556, 1.5556, 2.5556, 3.5556, 4.5556, 5.5556, 6.5556, 7.5556, 8.5556]])
torchsort.soft_sort(x, regularization_strength=0.1)
# tensor([[-0., 1., 2., 3., 5., 6., 7., 8., 9.]])

torchsort.soft_rank(x)
# tensor([[8., 1., 5., 4., 3., 2., 6., 7., 9.]])
```

Both operations are fully differentiable, on CPU or GPU:

```python
x = torch.tensor([[8., 0., 5., 3., 2., 1., 6., 7., 9.]], requires_grad=True).cuda()
y = torchsort.soft_sort(x)

torch.autograd.grad(y[0, 0], x)
# (tensor([[0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111]],
#         device='cuda:0'),)
```

## Example

### Spearman's Rank Coefficient

[Spearman's rank
coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
is a very useful metric for measuring how monotonically related two variables
are. We can use Torchsort to create a differentiable Spearman's rank coefficient
function so that we can optimize a model directly for this metric:

```python
import torch
import torchsort

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

pred = torch.tensor([[1., 2., 3., 4., 5.]], requires_grad=True)
target = torch.tensor([[5., 6., 7., 8., 7.]])
spearman = spearmanr(pred, target)
# tensor(0.8321)

torch.autograd.grad(spearman, pred)
# (tensor([[-5.5470e-02,  2.9802e-09,  5.5470e-02,  1.1094e-01, -1.1094e-01]]),)
```

## Benchmark

![Benchmark](https://github.com/teddykoker/torchsort/raw/main/extra/benchmark.png)

`torchsort` and `fast_soft_sort` each operate with a time complexity of *O(n log
n)*, each with some additional overhead when compared to the built-in
`torch.sort`. With a batch size of 1 (see left), the Numba JIT'd forward pass of
`fast_soft_sort` performs about on-par with the `torchsort` CPU kernel, however
its backward pass still relies on some Python code, which greatly penalizes its
performance. 

Furthermore, the `torchsort` kernel supports batches, and yields much better
performance than `fast_soft_sort` as the batch size increases.

![Benchmark](https://github.com/teddykoker/torchsort/raw/main/extra/benchmark_cuda.png)

The `torchsort` CUDA kernel performs quite well with sequence lengths under
~2000, and scales to extremely large batch sizes. In the future the
CUDA kernel can likely be further optimized to achieve performance closer to that of the
built in `torch.sort`.


## Reference

```bibtex
@inproceedings{blondel2020fast,
  title={Fast differentiable sorting and ranking},
  author={Blondel, Mathieu and Teboul, Olivier and Berthet, Quentin and Djolonga, Josip},
  booktitle={International Conference on Machine Learning},
  pages={950--959},
  year={2020},
  organization={PMLR}
}
```
