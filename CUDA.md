# Context
A [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context) represents all the relevant state that are required on an accelerator in order to instantiate and perform tasks. A Context includes data, variables, conditions, and more which define the environment in which the provided tasks are executed. Commands such as launching a kernel on a gpu are executed in a Context. Once a context is destroyed CUDA cleans up all the resources associated with it. Therefore, pointers originating from different contexts reference distinct address spaces (memory locations). Contexts are manages in a stack, each host (CPU) thread scheduling tasks has its own stack of contexts. Contexts can be exchanged between host threads. For instance, popping `ctx` from HostA and pushing it onto HostB will force operations executed from HostB to be executed in `ctx` while HostA will operate under the previous context in the stack. 

The context utilized for a device by the runtime API is the device's primary context. From the perspective of the runtime API a device and its primary context are synonymous.

# Module
Modules are dynamically loadable packages akin to DLLs or shared libraries. These include symbols, functions, and global variables that usres can call on. Modules maintain a module scope to avoid namespace collisions with other concurrently loaded modules.

# Hooks
Inheriting from `AcceleratorHooksInterface`, Hook implementations in PyTorch provide a generic interface through which host (CPU) code can query and set properties for the provided accelerators.

# CUDAStream
A stream is a structure that accepts events in a FIFO queue and executes them in a synchronous way, it can be thought of as a queue or pipeline for scheduling tasks on an accelerator. Spinning up multiple concurrent streams can enable task parallelism, for instance when we have multiple devices. In this case, each stream is uniquely associated with a device and queueing tasks to a stream will execute them on that device. Really, streams are specific to a context which are in-turn specific to a device. Streams have an associated integer priority, lower values are considered "high priority" by the accelerator's scheduling algorithm. 

CUDAStream abstracts the concept of a cuda stream (`cudaStream_t`), it maintains several pools of streams to reduce the overhead associated with common stream operations such as creation and destruction. Each device maintains 3 lazily intialized pools of streams, where the first pool contains the default stream. Pool 2 contains low priority streams. Pool 3 contains the high priority streams. Despite the fact that each thread in principle has its own "current stream," this stream pool is global across threads. Hence many host threads can potentially dispatch kernels and synchronize on the same stream. Synchronization can have [different meanings](https://leimao.github.io/blog/CUDA-Default-Stream/) depending on whether we are synchronizing to the legacy stream or via per-thread streams.

# CUDACachingAllocator
https://cs.stackexchange.com/questions/143650/difference-between-caching-and-slab-allocator
https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html