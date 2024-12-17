#include <ATen/zoom/hiprtc_stub/ATenHIPRTC.h>
#include <iostream>

namespace at { namespace zoom {

HIPRTC* load_hiprtc() {
  auto self = new HIPRTC();
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_HIPRTC(CREATE_ASSIGN)
  return self;
}

}} // at::zoom
