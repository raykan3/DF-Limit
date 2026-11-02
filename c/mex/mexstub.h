/* mexstub.h: declarations of the few functions of MATLAB's C Matrix and MEX APIs that dflimit_mex.c uses, so that the
 * MEX file can be compiled where no MATLAB is installed (the CI runners).  The names are the versioned symbols that
 * MATLAB's own headers map to since R2018a (the "800" API with size_t dimensions), the signatures those of the
 * documented functions.  With a MATLAB installation at hand, compile against its headers instead (without
 * -DDFLIMIT_STUB) to check that nothing has changed. */
#ifndef DFLIMIT_MEXSTUB_H
#define DFLIMIT_MEXSTUB_H
#include <stddef.h>
#include <stdbool.h>

typedef struct mxArray_tag mxArray;
typedef size_t mwSize;
typedef enum { mxREAL, mxCOMPLEX } mxComplexity;

#if defined(_WIN32)
#define MEX_IMPORT __declspec(dllimport)
#define MEX_EXPORT __declspec(dllexport)
#else
#define MEX_IMPORT
#define MEX_EXPORT __attribute__((visibility("default")))
#endif

MEX_IMPORT mxArray *mxCreateDoubleMatrix_800(mwSize m, mwSize n, mxComplexity flag);
MEX_IMPORT mxArray *mxCreateDoubleScalar_800(double value);
MEX_IMPORT mxArray *mxCreateString_800(const char *str);
MEX_IMPORT double *mxGetPr_800(const mxArray *pa);
MEX_IMPORT double mxGetScalar_800(const mxArray *pa);
MEX_IMPORT size_t mxGetNumberOfElements_800(const mxArray *pa);
MEX_IMPORT size_t mxGetM_800(const mxArray *pa);
MEX_IMPORT size_t mxGetN_800(const mxArray *pa);
MEX_IMPORT bool mxIsDouble_800(const mxArray *pa);
MEX_IMPORT bool mxIsComplex_800(const mxArray *pa);
MEX_IMPORT bool mxIsChar_800(const mxArray *pa);
MEX_IMPORT int mxGetString_800(const mxArray *pa, char *buf, mwSize buflen);
MEX_IMPORT char *mxArrayToString_800(const mxArray *pa);
MEX_IMPORT void mxFree_800(void *ptr);
MEX_IMPORT void mexErrMsgIdAndTxt_800(const char *identifier, const char *err_msg, ...);
MEX_IMPORT int mexCallMATLAB_800(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[], const char *fcn_name);
MEX_IMPORT void mexLock_800(void);

#define mxCreateDoubleMatrix mxCreateDoubleMatrix_800
#define mxCreateDoubleScalar mxCreateDoubleScalar_800
#define mxCreateString mxCreateString_800
#define mxGetPr mxGetPr_800
#define mxGetScalar mxGetScalar_800
#define mxGetNumberOfElements mxGetNumberOfElements_800
#define mxGetM mxGetM_800
#define mxGetN mxGetN_800
#define mxIsDouble mxIsDouble_800
#define mxIsComplex mxIsComplex_800
#define mxIsChar mxIsChar_800
#define mxGetString mxGetString_800
#define mxArrayToString mxArrayToString_800
#define mxFree mxFree_800
#define mexErrMsgIdAndTxt mexErrMsgIdAndTxt_800
#define mexCallMATLAB mexCallMATLAB_800
#define mexLock mexLock_800

/* the API version record that MATLAB's mex command compiles into every MEX file (extern/version/c_mexapi_version.c):
 * the 800 API, as built by its first release */
MEX_EXPORT void mexfilerequiredapiversion(unsigned int *built_by_rel, unsigned int *target_api_ver);
MEX_EXPORT void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
#define DFLIMIT_DEFINE_APIVERSION \
    void mexfilerequiredapiversion(unsigned int *built_by_rel, unsigned int *target_api_ver) { *built_by_rel = 0x2018a; *target_api_ver = 0x08000000; }
#endif
