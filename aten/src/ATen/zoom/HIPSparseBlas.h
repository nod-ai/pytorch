#pragma once

/*
  Provides a subset of hipSPARSE functions as templates:

    csrgeam2<scalar_t>(...)

  where scalar_t is double, float, c10::complex<double> or c10::complex<float>.
  The functions are available in at::zoom::sparse namespace.
*/

#include <ATen/zoom/ZoomContext.h>
#include <ATen/zoom/HIPSparse.h>

namespace at::zoom::sparse {

#define HIPSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(scalar_t)             \
  hipsparseHandle_t handle, int m, int n, const scalar_t *alpha,     \
      const hipsparseMatDescr_t descrA, int nnzA,                    \
      const scalar_t *csrSortedValA, const int *csrSortedRowPtrA,   \
      const int *csrSortedColIndA, const scalar_t *beta,            \
      const hipsparseMatDescr_t descrB, int nnzB,                    \
      const scalar_t *csrSortedValB, const int *csrSortedRowPtrB,   \
      const int *csrSortedColIndB, const hipsparseMatDescr_t descrC, \
      const scalar_t *csrSortedValC, const int *csrSortedRowPtrC,   \
      const int *csrSortedColIndC, size_t *pBufferSizeInBytes

template <typename scalar_t>
inline void csrgeam2_bufferSizeExt(
    HIPSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::csrgeam2_bufferSizeExt: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void csrgeam2_bufferSizeExt<float>(
    HIPSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(float));
template <>
void csrgeam2_bufferSizeExt<double>(
    HIPSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(double));
template <>
void csrgeam2_bufferSizeExt<c10::complex<float>>(
    HIPSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void csrgeam2_bufferSizeExt<c10::complex<double>>(
    HIPSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<double>));

#define CUSPARSE_CSRGEAM2_NNZ_ARGTYPES()                                      \
  hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA,     \
      int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,     \
      const hipsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB, \
      const int *csrSortedColIndB, const hipsparseMatDescr_t descrC,           \
      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace

template <typename scalar_t>
inline void csrgeam2Nnz(CUSPARSE_CSRGEAM2_NNZ_ARGTYPES()) {
  TORCH_HIPSPARSE_CHECK(hipsparseXcsrgeam2Nnz(
      handle,
      m,
      n,
      descrA,
      nnzA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      descrB,
      nnzB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedRowPtrC,
      nnzTotalDevHostPtr,
      workspace));
}

#define HIPSPARSE_CSRGEAM2_ARGTYPES(scalar_t)                                 \
  hipsparseHandle_t handle, int m, int n, const scalar_t *alpha,              \
      const hipsparseMatDescr_t descrA, int nnzA,                             \
      const scalar_t *csrSortedValA, const int *csrSortedRowPtrA,            \
      const int *csrSortedColIndA, const scalar_t *beta,                     \
      const hipsparseMatDescr_t descrB, int nnzB,                             \
      const scalar_t *csrSortedValB, const int *csrSortedRowPtrB,            \
      const int *csrSortedColIndB, const hipsparseMatDescr_t descrC,          \
      scalar_t *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, \
      void *pBuffer

template <typename scalar_t>
inline void csrgeam2(HIPSPARSE_CSRGEAM2_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::csrgeam2: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void csrgeam2<float>(HIPSPARSE_CSRGEAM2_ARGTYPES(float));
template <>
void csrgeam2<double>(HIPSPARSE_CSRGEAM2_ARGTYPES(double));
template <>
void csrgeam2<c10::complex<float>>(
    HIPSPARSE_CSRGEAM2_ARGTYPES(c10::complex<float>));
template <>
void csrgeam2<c10::complex<double>>(
    HIPSPARSE_CSRGEAM2_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRMM_ARGTYPES(scalar_t)                                    \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,                         \
      hipsparseOperation_t transA, hipsparseOperation_t transB, int mb, int n, \
      int kb, int nnzb, const scalar_t *alpha,                               \
      const hipsparseMatDescr_t descrA, const scalar_t *bsrValA,              \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      const scalar_t *B, int ldb, const scalar_t *beta, scalar_t *C, int ldc

template <typename scalar_t>
inline void bsrmm(HIPSPARSE_BSRMM_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrmm: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrmm<float>(HIPSPARSE_BSRMM_ARGTYPES(float));
template <>
void bsrmm<double>(HIPSPARSE_BSRMM_ARGTYPES(double));
template <>
void bsrmm<c10::complex<float>>(HIPSPARSE_BSRMM_ARGTYPES(c10::complex<float>));
template <>
void bsrmm<c10::complex<double>>(HIPSPARSE_BSRMM_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRMV_ARGTYPES(scalar_t)                                    \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,                         \
      hipsparseOperation_t transA, int mb, int nb, int nnzb,                  \
      const scalar_t *alpha, const hipsparseMatDescr_t descrA,                \
      const scalar_t *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, \
      int blockDim, const scalar_t *x, const scalar_t *beta, scalar_t *y

template <typename scalar_t>
inline void bsrmv(HIPSPARSE_BSRMV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrmv: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrmv<float>(HIPSPARSE_BSRMV_ARGTYPES(float));
template <>
void bsrmv<double>(HIPSPARSE_BSRMV_ARGTYPES(double));
template <>
void bsrmv<c10::complex<float>>(HIPSPARSE_BSRMV_ARGTYPES(c10::complex<float>));
template <>
void bsrmv<c10::complex<double>>(HIPSPARSE_BSRMV_ARGTYPES(c10::complex<double>));

#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

#define HIPSPARSE_BSRSV2_BUFFER_ARGTYPES(scalar_t)                 \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,              \
      hipsparseOperation_t transA, int mb, int nnzb,               \
      const hipsparseMatDescr_t descrA, scalar_t *bsrValA,         \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim, \
      bsrsv2Info_t info, int *pBufferSizeInBytes

template <typename scalar_t>
inline void bsrsv2_bufferSize(HIPSPARSE_BSRSV2_BUFFER_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrsv2_bufferSize: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsv2_bufferSize<float>(HIPSPARSE_BSRSV2_BUFFER_ARGTYPES(float));
template <>
void bsrsv2_bufferSize<double>(HIPSPARSE_BSRSV2_BUFFER_ARGTYPES(double));
template <>
void bsrsv2_bufferSize<c10::complex<float>>(
    HIPSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<float>));
template <>
void bsrsv2_bufferSize<c10::complex<double>>(
    HIPSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRSV2_ANALYSIS_ARGTYPES(scalar_t)               \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,              \
      hipsparseOperation_t transA, int mb, int nnzb,               \
      const hipsparseMatDescr_t descrA, const scalar_t *bsrValA,   \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim, \
      bsrsv2Info_t info, hipsparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsv2_analysis(HIPSPARSE_BSRSV2_ANALYSIS_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrsv2_analysis: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsv2_analysis<float>(HIPSPARSE_BSRSV2_ANALYSIS_ARGTYPES(float));
template <>
void bsrsv2_analysis<double>(HIPSPARSE_BSRSV2_ANALYSIS_ARGTYPES(double));
template <>
void bsrsv2_analysis<c10::complex<float>>(
    HIPSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<float>));
template <>
void bsrsv2_analysis<c10::complex<double>>(
    HIPSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRSV2_SOLVE_ARGTYPES(scalar_t)                           \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,                       \
      hipsparseOperation_t transA, int mb, int nnzb, const scalar_t *alpha, \
      const hipsparseMatDescr_t descrA, const scalar_t *bsrValA,            \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,          \
      bsrsv2Info_t info, const scalar_t *x, scalar_t *y,                   \
      hipsparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsv2_solve(HIPSPARSE_BSRSV2_SOLVE_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrsv2_solve: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsv2_solve<float>(HIPSPARSE_BSRSV2_SOLVE_ARGTYPES(float));
template <>
void bsrsv2_solve<double>(HIPSPARSE_BSRSV2_SOLVE_ARGTYPES(double));
template <>
void bsrsv2_solve<c10::complex<float>>(
    HIPSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<float>));
template <>
void bsrsv2_solve<c10::complex<double>>(
    HIPSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRSM2_BUFFER_ARGTYPES(scalar_t)                            \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,                         \
      hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int n, \
      int nnzb, const hipsparseMatDescr_t descrA, scalar_t *bsrValA,          \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      bsrsm2Info_t info, int *pBufferSizeInBytes

template <typename scalar_t>
inline void bsrsm2_bufferSize(HIPSPARSE_BSRSM2_BUFFER_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrsm2_bufferSize: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsm2_bufferSize<float>(HIPSPARSE_BSRSM2_BUFFER_ARGTYPES(float));
template <>
void bsrsm2_bufferSize<double>(HIPSPARSE_BSRSM2_BUFFER_ARGTYPES(double));
template <>
void bsrsm2_bufferSize<c10::complex<float>>(
    HIPSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<float>));
template <>
void bsrsm2_bufferSize<c10::complex<double>>(
    HIPSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRSM2_ANALYSIS_ARGTYPES(scalar_t)                          \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,                         \
      hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int n, \
      int nnzb, const hipsparseMatDescr_t descrA, const scalar_t *bsrValA,    \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      bsrsm2Info_t info, hipsparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsm2_analysis(HIPSPARSE_BSRSM2_ANALYSIS_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrsm2_analysis: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsm2_analysis<float>(HIPSPARSE_BSRSM2_ANALYSIS_ARGTYPES(float));
template <>
void bsrsm2_analysis<double>(HIPSPARSE_BSRSM2_ANALYSIS_ARGTYPES(double));
template <>
void bsrsm2_analysis<c10::complex<float>>(
    HIPSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<float>));
template <>
void bsrsm2_analysis<c10::complex<double>>(
    HIPSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<double>));

#define HIPSPARSE_BSRSM2_SOLVE_ARGTYPES(scalar_t)                             \
  hipsparseHandle_t handle, hipsparseDirection_t dirA,                         \
      hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int n, \
      int nnzb, const scalar_t *alpha, const hipsparseMatDescr_t descrA,      \
      const scalar_t *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, \
      int blockDim, bsrsm2Info_t info, const scalar_t *B, int ldb,           \
      scalar_t *X, int ldx, hipsparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsm2_solve(HIPSPARSE_BSRSM2_SOLVE_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::zoom::sparse::bsrsm2_solve: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsm2_solve<float>(HIPSPARSE_BSRSM2_SOLVE_ARGTYPES(float));
template <>
void bsrsm2_solve<double>(HIPSPARSE_BSRSM2_SOLVE_ARGTYPES(double));
template <>
void bsrsm2_solve<c10::complex<float>>(
    HIPSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<float>));
template <>
void bsrsm2_solve<c10::complex<double>>(
    HIPSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>));

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

} // namespace at::zoom::sparse
