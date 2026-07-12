from torchsort.ops import soft_rank, soft_sort

__all__ = ["soft_rank", "soft_sort", "softrank", "softsort"]

# Backward/compat aliases for callers that use camel/compact names.
softrank = soft_rank
softsort = soft_sort


def __getattr__(name):
    normalized = name.strip().lower().replace(" ", "_")
    if normalized in {"soft_rank", "softrank"}:
        return soft_rank
    if normalized in {"soft_sort", "softsort"}:
        return soft_sort
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
