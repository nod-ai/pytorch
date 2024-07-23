import torch
import torch._C
from typing import List


def get_amp_supported_dtype() -> List[torch.dtype]:
    return [torch.float16, torch.bfloat16, torch.float32]