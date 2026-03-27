#pragma once
#include <stdint.h>

// Auto-generated normalization params from ckpt
// NOTE: channel order MUST match training

// x_cols (order matters):
//   [ 0] Current
//   [ 1] dCurrent
//   [ 2] B
//   [ 3] dB
//   [ 4] Voltage
//   [ 5] CurrentSmallSig
//   [ 6] dCurrentSmallSig

// y_cols (order matters):
//   [ 0] AirGap

static const int TCN_QUANT_CIN  = 7;
static const float TCN_QUANT_NORM_EPS = 1.00000000e-12f;

static const float tcn_quant_x_min[7] = 
  {196.0, -21.0, 496.0, -14.0, 0.0, -111.0, -103.0};

static const float tcn_quant_x_max[7] = 
  {358.0, 20.0, 684.0, 11.0, 8000.0, 78.0, 99.0};

static const int TCN_QUANT_COUT = 1;

static const float tcn_quant_y_min[1] = 
  {527.0};

static const float tcn_quant_y_max[1] = 
  {772.0};

// ---- Helper functions (optional) ----
static inline float norm_01to11(float x, float x_min, float x_max, float eps) {
  float denom = x_max - x_min;
  if (denom < eps && denom > -eps) denom = 1.0f;
  float z = (x - x_min) / denom;
  return 2.0f * z - 1.0f;
}

static inline float denorm_11to01(float xn, float x_min, float x_max) {
  // inverse of norm_01to11
  float z = (xn + 1.0f) * 0.5f;
  return z * (x_max - x_min) + x_min;
}
