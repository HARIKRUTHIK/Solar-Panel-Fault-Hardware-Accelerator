// =============================================================================
// Module: zero_aware_mac
// Project: CNN Hardware Accelerator for Solar Panel Fault Classification
// Description: Energy-efficient Zero-Aware Multiply-Accumulate (MAC) tile
//              Detects zero operands in activations or weights, gating the
//              multiplier datapath to eliminate redundant dynamic switching power.
// =============================================================================

`timescale 1ns/1ps

module zero_aware_mac (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire        clear_acc,
    input  wire signed [7:0]  act_in,     // 8-bit signed quantized activation
    input  wire signed [7:0]  wt_in,      // 8-bit signed quantized weight
    output reg  signed [31:0] acc_out,    // 32-bit accumulated sum
    output reg  [31:0] total_mac_count,   // Tally of total MAC instructions
    output reg  [31:0] zero_skip_count,   // Tally of bypassed zero multiplications
    output wire        zero_detected      // High if act_in == 0 or wt_in == 0
);

    // 1. Zero Detection Logic (Combinational NOR reduction)
    wire act_is_zero = (act_in == 8'sd0);
    wire wt_is_zero  = (wt_in  == 8'sd0);
    assign zero_detected = act_is_zero | wt_is_zero;

    // 2. Gated Multiplier Operand Registers
    reg signed [7:0]  act_reg;
    reg signed [7:0]  wt_reg;
    reg               mult_enable;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_reg     <= 8'sd0;
            wt_reg      <= 8'sd0;
            mult_enable <= 1'b0;
        end else if (enable) begin
            if (!zero_detected) begin
                act_reg     <= act_in;
                wt_reg      <= wt_in;
                mult_enable <= 1'b1;
            end else begin
                // Hold inputs constant to eliminate dynamic capacitive switching (alpha * C * V^2 * f)
                mult_enable <= 1'b0;
            end
        end
    end

    // 3. Multiplier & Accumulator Stage
    wire signed [15:0] prod = act_reg * wt_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out         <= 32'sd0;
            total_mac_count <= 32'd0;
            zero_skip_count <= 32'd0;
        end else if (enable) begin
            if (clear_acc) begin
                acc_out <= 32'sd0;
            end else if (mult_enable) begin
                acc_out <= acc_out + {{16{prod[15]}}, prod};
            end

            // Performance Profiling Counters
            total_mac_count <= total_mac_count + 1'b1;
            if (zero_detected) begin
                zero_skip_count <= zero_skip_count + 1'b1;
            end
        end
    end

endmodule
