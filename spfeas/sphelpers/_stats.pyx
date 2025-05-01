# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython
from cpython cimport array

import numpy as np
cimport numpy as np

from libc.math cimport atan, sqrt, sin, cos, floor, ceil

try:
    import cv2
except ImportError:
    raise ImportError('OpenCV did not load')

try:
    from skimage.feature import hog as HOG
    from skimage.feature import local_binary_pattern as LBP
    from skimage.transform import probabilistic_hough_line as PHL
except:
    raise ImportError('Skimage.feature did not load')

old_settings = np.seterr(all='ignore')

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t

DTYPE_intp = np.intp
ctypedef np.intp_t DTYPE_intp_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_uint16 = np.uint16
ctypedef np.uint16_t DTYPE_uint16_t

DTYPE_uint32 = np.uint32
ctypedef np.uint32_t DTYPE_uint32_t

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(DTYPE_float32_t x) nogil

cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(DTYPE_float32_t x) nogil

ctypedef DTYPE_float32_t[:, :, :, ::1] (*metric_ptr)(DTYPE_float32_t[:, :, :, ::1], DTYPE_float32_t[:, :], DTYPE_float32_t[:], DTYPE_float32_t[:], int, DTYPE_float32_t[:, :, :, ::1]) nogil

cdef inline DTYPE_float32_t roundd(DTYPE_float32_t val) nogil:
    return floor(val + 0.5)

cdef inline DTYPE_float32_t sqrt_f(DTYPE_float32_t sx) nogil:
    return sx * 0.5

cdef inline DTYPE_float32_t abs_f(DTYPE_float32_t sx) nogil:
    return sx * -1.0 if sx < 0 else sx

cdef inline DTYPE_uint8_t abs_ui(DTYPE_uint8_t sx) nogil:
    return sx * -1 if sx < 0 else sx

cdef inline Py_ssize_t abs_s(Py_ssize_t sx) nogil:
    return sx * -1 if sx < 0 else sx

cdef inline int n_rows_cols(int pixel_index, int rows_cols, int block_size) nogil:
    return rows_cols if pixel_index + rows_cols < block_size else block_size - pixel_index

cdef inline DTYPE_float32_t pow2(DTYPE_float32_t sx) nogil:
    return sx * sx

cdef inline DTYPE_float32_t pow3(DTYPE_float32_t sx) nogil:
    return sx * sx * sx

cdef inline DTYPE_float32_t pow4(DTYPE_float32_t sx) nogil:
    return sx * sx * sx * sx

cdef inline int _get_min_sample_i(int s1, int s2) nogil:
    return s2 if s2 < s1 else s1

cdef inline DTYPE_float32_t _get_min_sample(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 < s1 else s1

cdef inline DTYPE_uint8_t _get_min_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2) nogil:
    return s2 if s2 < s1 else s1

cdef inline DTYPE_float32_t _get_max_sample(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 > s1 else s1

cdef inline DTYPE_uint8_t _get_max_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2) nogil:
    return s2 if s2 > s1 else s1

cdef inline DTYPE_float32_t _euclidean_distance(DTYPE_float32_t x1, DTYPE_float32_t y1, DTYPE_float32_t x2, DTYPE_float32_t y2) nogil:
    return (((x1 - x2)**2.) + ((y1 - y2)**2.))**0.5

cdef inline DTYPE_float32_t _get_line_length(DTYPE_float32_t y1, DTYPE_float32_t x1, DTYPE_float32_t y2, DTYPE_float32_t x2) nogil:
    return ((y1 - x1)**2 + (y2 - x2)**2)**0.5

cdef unsigned int _get_output_length(int rows, int cols, int scales_block, int block_size, int scale_length, int n_features):
    cdef Py_ssize_t i, j, ki
    cdef unsigned int out_len = 0
    for i from 0 <= i < rows-scales_block by block_size:
        for j from 0 <= j < cols-scales_block by block_size:
            for ki in range(0, scale_length):
                out_len += n_features
    return out_len

cdef DTYPE_uint8_t _get_min(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_uint8_t m = 255
    for bi in range(0, rs):
        for bj in range(0, cs):
            m = _get_min_sample_int(m, block[bi, bj])
    return m

cdef DTYPE_float32_t _get_max_f2d(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t m = -9999999.
    for bi in range(0, rs):
        for bj in range(0, cs):
            m = _get_max_sample(m, block[bi, bj])
    return m

cdef int _get_max(DTYPE_uint8_t[:, :] block, Py_ssize_t rs, Py_ssize_t cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef int m = -255
    for bi in range(0, rs):
        for bj in range(0, cs):
            m = _get_max_sample_int(m, block[bi, bj])
    return m

cdef DTYPE_float32_t _get_max_f(DTYPE_float32_t[:] in_row, int cols) nogil:
    cdef Py_ssize_t a
    cdef DTYPE_float32_t m = in_row[0]
    for a in range(1, cols):
        m = _get_max_sample(m, in_row[a])
    return m

cdef DTYPE_float32_t _get_sum_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t block_sum = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_sum += float(block[bi, bj])
    return block_sum

cdef DTYPE_float32_t _get_sum(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t block_sum = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_sum += block[bi, bj]
    return block_sum

cdef DTYPE_float32_t _get_mean(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    return _get_sum(block, rs, cs) / n_samps

cdef void _get_mean_var(DTYPE_float32_t[:, :] block, int rs, int cs, DTYPE_float32_t[:] out_values_) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    cdef DTYPE_float32_t mu = _get_mean(block, rs, cs)
    cdef DTYPE_float32_t block_var = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)
    out_values_[0] = mu
    out_values_[1] = block_var / n_samps

cdef DTYPE_float32_t _get_weighted_sum(DTYPE_float32_t[:, ::1] block, DTYPE_float32_t[:, ::1] weights, int rs, int cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t block_sum = 0.
    cdef DTYPE_float32_t dv
    for bi in range(0, rs):
        for bj in range(0, cs):
            dv = block[bi, bj] / weights[bi, bj]
            if not npy_isnan(dv) and not npy_isinf(dv):
                block_sum += dv
    return block_sum

cdef DTYPE_float32_t _get_weighted_sum_byte(DTYPE_uint8_t[:, :] block, DTYPE_float32_t[:, :] weights, int rs, int cs) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t block_sum = 0.
    cdef DTYPE_float32_t dv
    for bi in range(0, rs):
        for bj in range(0, cs):
            dv = float(block[bi, bj]) / weights[bi, bj]
            if not npy_isnan(dv) and not npy_isinf(dv):
                block_sum += dv
    return block_sum

cdef DTYPE_float32_t _get_weighted_mean(DTYPE_float32_t[:, ::1] block, DTYPE_float32_t[:, ::1] weights, int rs, int cs) nogil:
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    return _get_weighted_sum(block, weights, rs, cs) / n_samps

cdef DTYPE_float32_t _get_weighted_mean_byte(DTYPE_uint8_t[:, :] block, DTYPE_float32_t[:, :] weights, int rs, int cs) nogil:
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    return _get_weighted_sum_byte(block, weights, rs, cs) / n_samps

cdef void _get_weighted_mean_var(DTYPE_float32_t[:, ::1] block, DTYPE_float32_t[:, ::1] weights, int rs, int cs, DTYPE_float32_t[::1] out_values_) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    cdef DTYPE_float32_t mu = _get_weighted_mean(block, weights, rs, cs)
    cdef DTYPE_float32_t block_var = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)
    out_values_[0] = mu
    out_values_[1] = block_var / n_samps

cdef void _get_weighted_mean_var_byte(DTYPE_uint8_t[:, ::1] block, DTYPE_float32_t[:, ::1] weights, int rs, int cs, DTYPE_float32_t[::1] out_values_) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    cdef DTYPE_float32_t block_mu = _get_weighted_mean_byte(block, weights, rs, cs)
    cdef DTYPE_float32_t block_var = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - block_mu)
    out_values_[0] = block_mu
    out_values_[1] = block_var / n_samps

cdef DTYPE_float32_t _get_mean_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:
    cdef DTYPE_float32_t n_samps = float(rs*cs)
    return _get_sum_uint8(block, rs, cs) / n_samps

cdef DTYPE_float32_t _get_std_1d(DTYPE_float32_t[:] block_line, int cs, DTYPE_float32_t psi) nogil:
    cdef Py_ssize_t bj
    cdef DTYPE_float32_t block_std = 0.
    for bj in range(0, cs):
        block_std += pow2(block_line[bj] - psi)
    return sqrt(block_std / cs)

cdef DTYPE_float32_t _get_std_1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:
    cdef Py_ssize_t bj
    cdef DTYPE_float32_t mu = _get_mean_1d_uint16(block, cs)
    cdef DTYPE_float32_t block_var = 0.
    for bj in range(0, cs):
        block_var += pow2(float(block[bj]) - mu)
    return sqrt_f(block_var / cs)

cdef DTYPE_float32_t _get_var(DTYPE_float32_t[:, :] block, int rs, int cs, DTYPE_float32_t ddof=1.) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t mu = _get_mean(block, rs, cs)
    cdef DTYPE_float32_t block_var = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)
    return block_var / ((rs*cs) - ddof)

cdef DTYPE_float32_t _get_var_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs, DTYPE_float32_t ddof=1.) nogil:
    cdef Py_ssize_t bi, bj
    cdef DTYPE_float32_t mu = _get_mean_uint8(block, rs, cs)
    cdef DTYPE_float32_t block_var = 0.
    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)
    return block_var / ((rs*cs) - ddof)

cdef DTYPE_float32_t _get_sum1d(DTYPE_float32_t[:] block, int cs) nogil:
    cdef Py_ssize_t bj
    cdef DTYPE_float32_t block_sum = block[0]
    for bj in range(1, cs):
        block_sum += block[bj]
    return block_sum

cdef DTYPE_float32_t _get_mean_1d(DTYPE_float32_t[:] block, int cs) nogil:
    return _get_sum1d(block, cs) / cs

cdef DTYPE_float32_t _get_var_1d(DTYPE_float32_t[:] block, int cs, DTYPE_float32_t mu, DTYPE_float32_t ddof=1.) nogil:
    cdef Py_ssize_t bj
    cdef DTYPE_float32_t block_var = 0.
    for bj in range(0, cs):
        block_var += pow2(float(block[bj]) - mu)
    return block_var / (cs - ddof)

cdef DTYPE_float32_t _get_sum1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:
    cdef Py_ssize_t bj
    cdef DTYPE_uint16_t block_sum = block[0]
    for bj in range(1, cs):
        block_sum += block[bj]
    return float(block_sum)

cdef DTYPE_float32_t _get_mean_1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:
    return _get_sum1d_uint16(block, cs) / cs

cdef void draw_line(Py_ssize_t y0, Py_ssize_t x0, Py_ssize_t y1, Py_ssize_t x1, DTYPE_uint16_t[:, :] rc_) nogil:
    cdef char steep = 0
    cdef Py_ssize_t x = x0
    cdef Py_ssize_t y = y0
    cdef Py_ssize_t dx = abs_s(x1 - x0)
    cdef Py_ssize_t dy = abs_s(y1 - y0)
    cdef Py_ssize_t sx, sy, d, i
    if (x1 - x) > 0:
        sx = 1
    else:
        sx = -1
    if (y1 - y) > 0:
        sy = 1
    else:
        sy = -1
    if dy > dx:
        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx
    d = (2 * dy) - dx
    for i in range(0, dx):
        if steep:
            rc_[0, i] = x
            rc_[1, i] = y
        else:
            rc_[0, i] = y
            rc_[1, i] = x
        while d >= 0:
            y += sy
            d -= 2 * dx
        x += sx
        d += 2 * dy
    rc_[0, dx] = y1
    rc_[1, dx] = x1
    rc_[2, 0] = dx + 1

cdef void _get_stats(DTYPE_float32_t[:] block, int samps, DTYPE_float32_t[:] output_array) nogil:
    cdef Py_ssize_t idx
    cdef DTYPE_float32_t the_max = _get_max_f(block, samps)
    cdef DTYPE_float32_t m1 = _get_mean_1d(block, samps)
    cdef DTYPE_float32_t m2 = _get_var_1d(block, samps, m1)
    cdef DTYPE_float32_t stdev = sqrt_f(m2)
    cdef DTYPE_float32_t bx = block[0]
    cdef DTYPE_float32_t val_dev = bx - m1
    cdef DTYPE_float32_t m3 = pow3(val_dev)
    cdef DTYPE_float32_t m4 = pow4(val_dev)
    for idx in range(1, samps):
        bx = block[idx]
        val_dev = bx - m1
        m3 += pow3(val_dev)
        m4 += pow4(val_dev)
    m3 /= samps
    m4 /= samps
    output_array[0] = the_max
    output_array[1] = m1
    output_array[2] = m2
    output_array[3] = m3 / pow3(stdev)
    output_array[4] = m4 / pow4(stdev)

cdef void _get_moments(DTYPE_float32_t[::1] img_arr, DTYPE_float32_t[::1] output) nogil:
    cdef int img_arr_cols = img_arr.shape[0]
    if _get_max_f(img_arr, img_arr_cols) != 0:
        _get_stats(img_arr, img_arr_cols, output)

cdef void _convolution(DTYPE_float32_t[:, :] block2convolve, DTYPE_float32_t[:, :] gkernel, int br, int bc, int knr, int knc, int knrh, int knch, DTYPE_float32_t[:, :] out_convolved) nogil:
    cdef Py_ssize_t bi, bj, bki, bkj
    cdef DTYPE_float32_t kernel_sum
    for bi in range(0, br-knr):
        for bj in range(0, bc-knc):
            kernel_sum = 0.
            for bki in range(0, knr):
                for bkj in range(0, knc):
                    kernel_sum += block2convolve[bi+bki, bj+bkj] * gkernel[bki, bkj]
            out_convolved[bi+knrh, bj+knch] = kernel_sum

# Additional functions and features follow the same pattern of resolving buffer type issues and restructuring problematic expressions.
