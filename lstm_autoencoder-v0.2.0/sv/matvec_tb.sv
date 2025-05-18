`timescale 1ns/1ps

module matvec_tb;

    // Parameters
    localparam ROWS = 4;
    localparam COLS = 4;
    localparam BANDWIDTH = 4;
    localparam DATA_WIDTH = 16;
    localparam ADDR_WIDTH = 12;

    // Clock and reset
    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    // Test signals
    logic start;
    logic [$clog2(ROWS):0] num_rows = ROWS;
    logic [$clog2(COLS):0] num_cols = COLS;
    logic vector_write_enable;
    logic [$clog2(COLS)-1:0] vector_base_addr;
    logic signed [DATA_WIDTH-1:0] vector_in [0:BANDWIDTH-1];

    logic [$clog2(ROWS*COLS)-1:0] matrix_addr;
    logic matrix_enable;
    logic signed [DATA_WIDTH-1:0] matrix_data [0:BANDWIDTH-1];
    logic matrix_ready;

    logic signed [(DATA_WIDTH*2)-1:0] result_out;
    logic result_valid;
    logic busy;

    // DUT instantiation
    matvec_multiplier #(
        .MAX_ROWS(ROWS),
        .MAX_COLS(COLS),
        .BANDWIDTH(BANDWIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .num_rows(num_rows),
        .num_cols(num_cols),
        .vector_write_enable(vector_write_enable),
        .vector_base_addr(vector_base_addr),
        .vector_in(vector_in),
        .matrix_addr(matrix_addr),
        .matrix_enable(matrix_enable),
        .matrix_data(matrix_data),
        .matrix_ready(matrix_ready),
        .result_out(result_out),
        .result_valid(result_valid),
        .busy(busy)
    );

    // Matrix loader
    matrix_loader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANDWIDTH(BANDWIDTH)
    ) loader (
        .clk(clk),
        .rst_n(rst_n),
        .enable(matrix_enable),
        .matrix_addr(matrix_addr),
        .matrix_data(matrix_data),
        .ready(matrix_ready)
    );

    // Test vector (Q4.12 format)
    logic signed [DATA_WIDTH-1:0] test_vector [0:COLS-1];

    // Expected result (Q4.12 format)
    logic signed [DATA_WIDTH-1:0] expected_result [0:ROWS-1];
    integer pass_count = 0;

    initial begin
        automatic int row = 0;
        // Reset
        rst_n = 0;
        start = 0;
        vector_write_enable = 0;
        #20;
        rst_n = 1;
        #20;

        // Define test vector: [1, 2, 3, 4] in Q4.12
        test_vector[0] = 16'sd4096;   // 1.0
        test_vector[1] = 16'sd8192;   // 2.0
        test_vector[2] = 16'sd12288;  // 3.0
        test_vector[3] = 16'sd16384;  // 4.0

        // Start operation
        start = 1;
        @(posedge clk);
        start = 0;

        // Load test vector into DUT
        for (int i = 0; i < COLS; i += BANDWIDTH) begin
            vector_base_addr = i;
            for (int j = 0; j < BANDWIDTH; j++) begin
                vector_in[j] = test_vector[i + j];
            end
            vector_write_enable = 1;
            @(posedge clk);
            // TODO: Is it necessary to drop this signal between chunks?
            // Seems like it would make matvec_mul switch between idle and
            // vector load states, which does not seem like it's quite what
            // we want, even though it may work
            vector_write_enable = 0;
            @(posedge clk);
        end

        // Manually define expected result from known matrix (ramp: [0...15])
        // Each row of the matrix is:
        // [0 1 2 3] → dot [1 2 3 4] = 0*1 + 1*2 + 2*3 + 3*4 = 20 → 20.0 in Q4.12
        // [4 5 6 7] → dot [1 2 3 4] = 4*1 + 5*2 + 6*3 + 7*4 = 56 → 56.0
        // [8 9 10 11] = 92.0, [12 13 14 15] = 128.0
        expected_result[0] = 16'sd81920;   // 20.0 * 4096 = 81920
        expected_result[1] = 16'sd229376;  // 56.0
        expected_result[2] = 16'sd376832;  // 92.0
        expected_result[3] = 16'sd524288;  // 128.0

        @(posedge matrix_ready);

        // Wait and check results
        while (row < ROWS) begin
            @(posedge clk);
            if (result_valid) begin
                $display("Row %0d: Got %0d, Expected %0d", row, result_out, expected_result[row]);
                if (result_out === expected_result[row]) begin
                    $display("✅ PASS");
                    pass_count++;
                end else begin
                    $display("❌ FAIL");
                end
            end
            row++;
        end

        $display("Test complete: %0d/%0d passed", pass_count, ROWS);
        $finish;
    end

endmodule
