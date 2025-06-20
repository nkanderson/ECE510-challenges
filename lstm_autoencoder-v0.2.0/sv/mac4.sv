//------------------------------------------------------------------------------
// File      : mac4.sv
// Author    : Niklas Anderson with ChatGPT assistance
// Project   : ECE510 Challenge 10
// Created   : Spring 2025
// Description : 4-input MAC (Multiply-Accumulate) module
//------------------------------------------------------------------------------

module mac4 #(
    parameter DATA_WIDTH = 16
)(
    // clk signal is for synthesis
    input  logic clk,
    input  logic signed [DATA_WIDTH-1:0] a0, b0,
    input  logic signed [DATA_WIDTH-1:0] a1, b1,
    input  logic signed [DATA_WIDTH-1:0] a2, b2,
    input  logic signed [DATA_WIDTH-1:0] a3, b3,
    output logic signed [(DATA_WIDTH*2)-1:0] result  // Q4.12
);

    localparam RESULT_WIDTH = 2*DATA_WIDTH;

    // Internal products: Q6.26
    logic signed [RESULT_WIDTH-1:0] p0, p1, p2, p3;
    assign p0 = a0 * b0;
    assign p1 = a1 * b1;
    assign p2 = a2 * b2;
    assign p3 = a3 * b3;

    // Shifted to Q4.12
    logic signed [RESULT_WIDTH-1:0] s0, s1, s2, s3;
    assign s0 = p0 >>> 14;
    assign s1 = p1 >>> 14;
    assign s2 = p2 >>> 14;
    assign s3 = p3 >>> 14;

    // Accumulate
    logic signed [RESULT_WIDTH-1:0] sum;
    assign sum = s0 + s1 + s2 + s3;

    // Final output - lower 16 bits contain Q4.12, returning 32-bit
    // value to account for any possible overflow
    assign result = sum[RESULT_WIDTH-1:0];

endmodule
