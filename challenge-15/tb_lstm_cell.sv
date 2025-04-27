`timescale 1ns/1ps

module tb_lstm_cell();

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

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock

    // Instantiate the DUT
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

    // Stimulus
    initial begin
        // Reset
        rst_n = 0;
        start = 0;
        repeat(2) @(posedge clk);
        rst_n = 1;

        // Initialize inputs
        x[0] = 32'sd1;
        x[1] = 32'sd2;

        h_prev[0] = 32'sd0;
        h_prev[1] = 32'sd0;

        c_prev[0] = 32'sd0;
        c_prev[1] = 32'sd0;

        // Initialize weights and biases
        foreach (W[i, j])
            W[i][j] = 32'sd1;
        foreach (U[i, j])
            U[i][j] = 32'sd1;
        foreach (b[i])
            b[i] = 32'sd0;

        // Start LSTM computation
        repeat(1) @(posedge clk);
        start = 1;
        repeat(1) @(posedge clk);
        start = 0;

        // Wait for computation to complete
        repeat(10) @(posedge clk);

        // Display results
        $display("\nFinal h outputs:");
        foreach (h[i])
            $display("h[%0d] = %0d", i, h[i]);

        $display("\nFinal c outputs:");
        foreach (c[i])
            $display("c[%0d] = %0d", i, c[i]);

        // Check results
        if (h[0] !== 32'sd1 || h[1] !== 32'sd1 || c[0] !== 32'sd1 || c[1] !== 32'sd1) begin
            $display("\nTEST FAILED: Unexpected output!");
            $stop;
        end else begin
            $display("\nTEST PASSED!");
        end

        $finish;
    end

endmodule
