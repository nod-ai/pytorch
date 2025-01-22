#define AT_ROCM_ENABLED() true
#define AT_MAGMA_ENABLED() false

// disabled for now because we're testing on an old hipsparselt
#ifdef HIPSPARSELT_ENABLED
#define AT_HIPSPARSELT_ENABLED() true
#else
#define AT_HIPSPARSELT_ENABLED() false
#endif