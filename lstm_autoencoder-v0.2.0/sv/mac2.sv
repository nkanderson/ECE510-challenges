//------------------------------------------------------------------------------
// File      : mac2.sv
// Author    : Niklas Anderson with ChatGPT assistance
// Project   : ECE510 Challenge 10
// Created   : Spring 2025
// Description : 2-input MAC (Multiply-Accumulate) module
//------------------------------------------------------------------------------

module mac2 #(
    parameter DATA_WIDTH = 16
)(
    input  logic signed [DATA_WIDTH-1:0] a0, b0,
    input  logic signed [DATA_WIDTH-1:0] a1, b1,
    output logic signed [(DATA_WIDTH*2)-1:0] result  // Q4.12
);

    localparam RESULT_WIDTH = 2*DATA_WIDTH;

    // Internal products: Q6.26
    logic signed [RESULT_WIDTH-1:0] p0, p1;
    assign p0 = a0 * b0;
    assign p1 = a1 * b1;

    // Shifted to Q4.12
    logic signed [RESULT_WIDTH-1:0] s0, s1;
    assign s0 = p0 >>> 14;
    assign s1 = p1 >>> 14;

    // Accumulate
    logic signed [RESULT_WIDTH-1:0] sum;
    assign sum = s0 + s1;

    // Final output - lower 16 bits contain Q4.12, returning 32-bit
    // value to account for any possible overflow
    assign result = sum[RESULT_WIDTH-1:0];

endmodule
