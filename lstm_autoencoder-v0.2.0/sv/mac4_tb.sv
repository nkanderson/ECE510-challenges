//------------------------------------------------------------------------------
// File      : mac4_tb.sv
// Author    : Niklas Anderson with ChatGPT assistance
// Project   : ECE510 Challenge 10
// Created   : Spring 2025
// Description : Testbench for the 4-input MAC (Multiply-Accumulate) module
//------------------------------------------------------------------------------

`timescale 1ns / 1ps

module mac4_tb;

    logic signed [15:0] a0, b0;
    logic signed [15:0] a1, b1;
    logic signed [15:0] a2, b2;
    logic signed [15:0] a3, b3;
    logic signed [31:0] result;

    // Instantiate DUT
    mac4 dut (
        .a0(a0), .b0(b0),
        .a1(a1), .b1(b1),
        .a2(a2), .b2(b2),
        .a3(a3), .b3(b3),
        .result(result)
    );

    initial begin
        // Test case 1 - Base case
        // Each product will be: (Q2.14 * Q4.12) = Q6.26 → >>>14 → Q4.12
        //
        // a0 = 0.5  (Q2.14) =  0b_0010_0000_0000_0000 = 8192
        // b0 = 2.0  (Q4.12) =  0b_0010_0000_0000_0000 = 8192
        // a1 = 1.0  (Q2.14) =  0b_0100_0000_0000_0000 = 16384
        // b1 = 1.0  (Q4.12) =  0b_0001_0000_0000_0000 = 4096
        // a2 = 1.5  (Q2.14) =  0b_0110_0000_0000_0000 = 6144
        // b2 = 2.0  (Q4.12) =  0b_0010_0000_0000_0000 = 8192
        // a3 = 0.25 (Q2.14) =  0b_0001_0000_0000_0000 = 4096
        // b3 = 4.0  (Q4.12) =  0b_0100_0000_0000_0000 = 16384
        //
        // Products (float):
        // 0.5 * 2.0   = 1.0
        // 1.0 * 1.0   = 1.0
        // 1.5 * 2.0   = 3.0
        // 0.25 * 4.0  = 1.0
        // Sum = 6

        a0 = 16'b0010_0000_0000_0000; //  0.5
        b0 = 16'b0010_0000_0000_0000; //  2.0

        a1 = 16'b0100_0000_0000_0000; //  1.0
        b1 = 16'b0001_0000_0000_0000; //  1.0

        a2 = 16'b0110_0000_0000_0000; //  1.5
        b2 = 16'b0010_0000_0000_0000; //  2.0

        a3 = 16'b0001_0000_0000_0000; //  0.25
        b3 = 16'b0100_0000_0000_0000; //  4.0

        #1;

        // Expect 6 in Q4.12 = 6 * 4096 = 24576 = 16'b0110_0000_0000_0000
        // with sign extension to make a 32-bit value
        if (result === 32'b0000_0000_0000_0000_0110_0000_0000_0000) begin
            $display("✅ PASS: result = %0d (Q20.12)", result);
        end else begin
            $display("❌ FAIL: result = %0d, expected 10240", result);
        end

        #5;

        // Test case 2 - Negative numbers in 2s complement
        // Products (float):
        //  0.5 *  4.0    =  2.0
        // -1.5 *  1.0    = -1.5
        //  1.0 * -2.25   = -2.25
        //  0.25 * 4.0    =  1.0
        //  Sum = -0.75
        a0 = 16'b0010_0000_0000_0000; //  0.5
        b0 = 16'b0100_0000_0000_0000; //  4.0

        a1 = 16'b1010_0000_0000_0000; //  -1.5
        b1 = 16'b0001_0000_0000_0000; //   1.0

        a2 = 16'b0100_0000_0000_0000; //   1.0
        b2 = 16'b1101_1100_0000_0000; //  -2.25

        a3 = 16'b0001_0000_0000_0000; //  0.25
        b3 = 16'b0100_0000_0000_0000; //  4.0

        #5;

        // Expect -0.75 in Q4.12 = -0.75 * 4096 = -3072 = 16'b1111_0100_0000_0000
        // with sign extension to make a 32-bit value
        if (result === 32'b1111_1111_1111_1111_1111_0100_0000_0000) begin
            $display("✅ PASS: result = %0d (Q20.12)", result);
        end else begin
            $display("❌ FAIL: result = %0d, expected -3072", result);
        end

        // Test case 3 - Overflow for 16 bit Q4.12
        // Products (float):
        //  1.5 *  4.0    =  6.0
        //  1.5 *  4.0    =  6.0
        //  1.5 *  4.0    =  6.0
        //  1.5 *  4.0    =  6.0
        //  Sum = 24.0
        a0 = 16'b0110_0000_0000_0000; //  1.5
        b0 = 16'b0100_0000_0000_0000; //  4.0

        a1 = 16'b0110_0000_0000_0000; //  1.5
        b1 = 16'b0100_0000_0000_0000; //  4.0

        a2 = 16'b0110_0000_0000_0000; //  1.5
        b2 = 16'b0100_0000_0000_0000; //  4.0

        a3 = 16'b0110_0000_0000_0000; //  1.5
        b3 = 16'b0100_0000_0000_0000; //  4.0

        #1;

        // Expect overflow 24 in Q4.12 = 24 * 4096 = 98304 = 17'b1_1000_0000_0000_0000
        // with sign extension to create 32 bit value
        if (result === 32'b0000_0000_0000_0001_1000_0000_0000_0000) begin
            $display("✅ PASS: result = %0d (Q20.12)", result);
        end else begin
            $display("❌ FAIL: result = %0d, expected 98304", result);
        end

        $finish;
    end

endmodule
