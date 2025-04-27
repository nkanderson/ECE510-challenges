// Testbench for lstm_cell
module tb_lstm_cell;

    parameter int INPUT_SIZE = 2;
    parameter int HIDDEN_SIZE = 2;
    parameter int DATA_WIDTH = 32;

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

    // Instantiate DUT
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

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Reset and stimulus
    initial begin
        integer i, j;
        rst_n = 0;
        start = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Initialize inputs
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

        $finish;
    end

endmodule
