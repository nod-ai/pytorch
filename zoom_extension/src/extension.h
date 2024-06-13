#pragma once
#include <torch/library.h>
#include <ATen/DeviceGuard.h>
#include "ZoomAllocator.h"
#include <ATen/native/cpu/Loops.h>
#include <c10/core/TensorOptions.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ops/empty.h>
#include <iostream>
#include <torch/csrc/Device.h>
#include <torch/extension.h>