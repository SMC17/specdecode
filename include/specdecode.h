#ifndef SPECDECODE_H
#define SPECDECODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Mask a float32 logit array in-place using a packed uint64_t bitmask.
 *
 * Bit `i` in `mask_words[i >> 6]` (LSB-first: `(mask_words[i >> 6] >> (i & 63)) & 1`)
 * indicates whether logit `i` is allowed (1) or masked to -INFINITY (0).
 *
 * @param logits      Pointer to `n_logits` float32 values (modified in-place).
 * @param n_logits    Number of logits in `logits`.
 * @param mask_words  Pointer to `n_words` uint64_t bitmask words.
 * @param n_words     Number of 64-bit words in `mask_words`.
 * @return            0 on success, -1 if `logits` or `mask_words` is NULL.
 */
int32_t specdecode_mask_logits_f32(
    float *logits,
    size_t n_logits,
    const uint64_t *mask_words,
    size_t n_words
);

#ifdef __cplusplus
}
#endif

#endif /* SPECDECODE_H */
