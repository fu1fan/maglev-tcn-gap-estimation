#pragma once
#include <stdint.h>

// Auto-generated from quant_report.json
// pow2 scale = 2^exp2 (use shifts)

static const int TCN_QUANT_W_BITS = 12;
static const int TCN_QUANT_A_BITS = 14;

// ---- Weight exp2 table ----
static const int16_t head_weight_exp2_w = -12;
static const int16_t tcn_0_conv1_weight_exp2_w = -8;
static const int16_t tcn_0_conv2_weight_exp2_w = -8;
static const int16_t tcn_0_downsample_weight_exp2_w = -9;
static const int16_t tcn_1_conv1_weight_exp2_w = -7;
static const int16_t tcn_1_conv2_weight_exp2_w = -9;
static const int16_t tcn_2_conv1_weight_exp2_w = -7;
static const int16_t tcn_2_conv2_weight_exp2_w = -8;
static const int16_t tcn_3_conv1_weight_exp2_w = -8;
static const int16_t tcn_3_conv2_weight_exp2_w = -8;
static const int16_t tcn_4_conv1_weight_exp2_w = -8;
static const int16_t tcn_4_conv2_weight_exp2_w = -7;

// ---- Bias exp2 table ----
static const int16_t head_bias_exp2_b = -21;
static const int16_t tcn_0_conv1_bias_exp2_b = -20;
static const int16_t tcn_0_conv2_bias_exp2_b = -18;
static const int16_t tcn_0_downsample_bias_exp2_b = -21;
static const int16_t tcn_1_conv1_bias_exp2_b = -18;
static const int16_t tcn_1_conv2_bias_exp2_b = -19;
static const int16_t tcn_2_conv1_bias_exp2_b = -17;
static const int16_t tcn_2_conv2_bias_exp2_b = -18;
static const int16_t tcn_3_conv1_bias_exp2_b = -18;
static const int16_t tcn_3_conv2_bias_exp2_b = -17;
static const int16_t tcn_4_conv1_bias_exp2_b = -18;
static const int16_t tcn_4_conv2_bias_exp2_b = -16;

// ---- Activation exp2 (suggested) ----
static const int16_t act_head_out_exp2 = -13;
static const int16_t act_input_exp2 = -12;
static const int16_t act_tcn_0_act1_out_exp2 = -10;
static const int16_t act_tcn_0_act2_out_exp2 = -11;
static const int16_t act_tcn_0_add_out_exp2 = -11;
static const int16_t act_tcn_0_final_out_exp2 = -11;
static const int16_t act_tcn_0_res_out_exp2 = -11;
static const int16_t act_tcn_1_act1_out_exp2 = -10;
static const int16_t act_tcn_1_act2_out_exp2 = -10;
static const int16_t act_tcn_1_add_out_exp2 = -10;
static const int16_t act_tcn_1_final_out_exp2 = -10;
static const int16_t act_tcn_1_res_out_exp2 = -11;
static const int16_t act_tcn_2_act1_out_exp2 = -10;
static const int16_t act_tcn_2_act2_out_exp2 = -11;
static const int16_t act_tcn_2_add_out_exp2 = -10;
static const int16_t act_tcn_2_final_out_exp2 = -10;
static const int16_t act_tcn_2_res_out_exp2 = -10;
static const int16_t act_tcn_3_act1_out_exp2 = -9;
static const int16_t act_tcn_3_act2_out_exp2 = -10;
static const int16_t act_tcn_3_add_out_exp2 = -10;
static const int16_t act_tcn_3_final_out_exp2 = -10;
static const int16_t act_tcn_3_res_out_exp2 = -10;
static const int16_t act_tcn_4_act1_out_exp2 = -9;
static const int16_t act_tcn_4_act2_out_exp2 = -9;
static const int16_t act_tcn_4_add_out_exp2 = -9;
static const int16_t act_tcn_4_final_out_exp2 = -9;
static const int16_t act_tcn_4_res_out_exp2 = -10;

