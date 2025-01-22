import torch.zoom

torch.utils.rename_privateuse1_backend("zoom")
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(
    unsupported_dtype=unsupported_dtypes
)
x = torch.empty(5, device="zoom:0", dtype=torch.int64)
print(x)
