# Setup Python Env

To start out, we just need to follow the normal procedure to build PyTorch from source. For convenience I've included these steps here:

```bash
conda create -n nod-pytorch python==3.10
conda activate nod-pytorch
conda install cmake ninja
pip install -r requirements.txt
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

# CMake Build

Using the `USE_ZOOM` flag with CMake will enable building with HIP for ROCm without requiring any of the "HIPify" scripts in order to build. This will include HIP libraries and populate `torch.version.hip` appropriately. This flag is NOT yet entered into the `setup.py` script, so for now it needs to be added manually via `cmake` or `ccmake`.

You'll need to set the `ROCM_PATH` and `HIP_ROOT_DIR` environment variables appropriately, by default on linux these should be `/opt/rocm/` and `/opt/rocm/hip` respectively.

If you're running on Linux you can just use `build.sh` to build:
```bash
cd pytorch/
source build.sh
```

Alternatively, if you want to manually setup your CMake build you can use the following commands:

```bash
cd build/
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_PATH=/opt/rocm
export HIP_ROOT_DIR=/opt/rocm/hip
cmake -DUSE_ZOOM=ON --build . --target install
```

# Running PyTorch with Zoom

Programs using the zoom backend must be prefaced with this stub until we register a proper dispatch key in pytorch

```python
import torch
import torch.zoom
torch.utils.rename_privateuse1_backend('zoom')
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=None)
```

# Installing Triton

Since main Triton currently treats ROCm as if its masquerading as `torch.cuda`, we need a custom installation:

```bash
git clone https://github.com/123epsilon/triton.git
cd triton/
git checkout zoom
pip install pybind11
pip install python/
```

# Running LLama3 with Triton using LigerKernels and HuggingFace

```bash
pip install liger-kernel
```

```python
# Run Llama 3
import torch
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from time import perf_counter as pf
torch.utils.rename_privateuse1_backend('zoom')

# Set up the model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="zoom"
)

# Function to generate text
def generate_text(prompt, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Hey, how are you doing today?"
s = pf()
response = generate_text(prompt)
e = pf()
print(f"Prompt: {prompt}")
print(f"Response: {response}")

print(f"{e-s} seconds")
```

```python
# Or run the instruct-tuned variant
import torch
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama
torch.utils.rename_privateuse1_backend('zoom')

apply_liger_kernel_to_llama()
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="zoom",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=30,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])

```