#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Returns zero when accepted into the bounded queue, -1 on null/empty/oversize
 * input, -2 on queue exhaustion, and -3 if unavailable. JSON parsing happens on
 * the writer; malformed accepted input becomes an explicit Loss record.
 * No exceptions cross ABI. */
int reqscope_emit_json(const char *record);
int reqscope_push_step(uint64_t step_id);
int reqscope_pop_step(void);
int reqscope_flush(void);
/* CUDA_INJECTION64_PATH entry point: CUDA expects 1 for success. */
int InitializeInjection(void);
#ifdef __cplusplus
}
#endif
