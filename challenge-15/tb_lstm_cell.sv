`timescale 1ns / 1ps

module tb_lstm_cell;

    parameter int INPUT_SIZE  = 2;
    parameter int HIDDEN_SIZE = 2;
    parameter int DATA_WIDTH  = 32;

    logic clk;
    logic rst_n;
    logic start;
    logic done;

    logic signed [DATA_WIDTH-1:0] x[INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] h_prev[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] c_prev[HIDDEN_SIZE];

    logic signed [DATA_WIDTH-1:0] W[HIDDEN_SIZE*4][INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] U[HIDDEN_SIZE*4][HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] b[HIDDEN_SIZE*4];

    logic signed [DATA_WIDTH-1:0] h[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] c[HIDDEN_SIZE];

    // DUT
    lstm_cell #(
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .x(x),
        .h_prev(h_prev),
        .c_prev(c_prev),
        .W(W),
        .U(U),
        .b(b),
        .h(h),
        .c(c)
    );

    // Clock
    always #5 clk = ~clk;

    initial begin
        integer i, j;

        // Initialize
        clk   = 0;
        rst_n = 0;
        start = 0;

        // Reset
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Init inputs
        for (i = 0; i < INPUT_SIZE; i++) x[i] = i + 1;
        for (i = 0; i < HIDDEN_SIZE; i++) h_prev[i] = i + 1;
        for (i = 0; i < HIDDEN_SIZE; i++) c_prev[i] = i + 1;

        for (i = 0; i < HIDDEN_SIZE*4; i++) begin
            for (j = 0; j < INPUT_SIZE; j++) W[i][j] = i + j + 1;
        end

        for (i = 0; i < HIDDEN_SIZE*4; i++) begin
            for (j = 0; j < HIDDEN_SIZE; j++) U[i][j] = i + j + 1;
        end

        for (i = 0; i < HIDDEN_SIZE*4; i++) b[i] = i + 1;

        // Start computation
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Wait for done
        wait (done);
        @(posedge clk);

        $display("Testbench completed. Output h and c values:");
        for (i = 0; i < HIDDEN_SIZE; i++) begin
            $display("h[%0d] = %0d", i, h[i]);
            $display("c[%0d] = %0d", i, c[i]);
        end

        // --- Self-check against expected values ---
        logic signed [DATA_WIDTH-1:0] expected_h[HIDDEN_SIZE];
        logic signed [DATA_WIDTH-1:0] expected_c[HIDDEN_SIZE];

        // TODO: Replace these with golden values once known
        expected_h[0] = 0;
        expected_h[1] = 0;
        expected_c[0] = 1310804;
        expected_c[1] = 1573064;

        for (i = 0; i < HIDDEN_SIZE; i++) begin
            if (h[i] !== expected_h[i]) begin
                $fatal(1, "FAIL: h[%0d] = %0d, expected %0d", i, h[i], expected_h[i]);
            end
            if (c[i] !== expected_c[i]) begin
                $fatal(1, "FAIL: c[%0d] = %0d, expected %0d", i, c[i], expected_c[i]);
            end
        end

        $display("✅ All checks passed.");
        $finish;
    end

endmodule
