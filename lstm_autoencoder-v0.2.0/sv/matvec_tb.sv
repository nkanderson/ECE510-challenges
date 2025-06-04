`timescale 1ns/1ps

module matvec_tb;

    // Parameters
    localparam MIN_ROWS = 4;
    localparam MIN_COLS = 4;
    localparam HALF_ROWS = 32;
    localparam HALF_COLS = 32;
    localparam MAX_ROWS = 64;
    localparam MAX_COLS = 64;
    localparam BANDWIDTH_4X4 = 4;
    localparam MAX_BANDWIDTH = 16;
    localparam VEC_NUM_BANKS_4X4  = 1;
    localparam VEC_NUM_BANKS  = 4;
    localparam DATA_WIDTH = 16;
    localparam ADDR_WIDTH = 12;

    //
    // === Signals ===
    //
    // Clock and reset
    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    // DUT 4x4 signals
    logic start_4x4;
    logic [$clog2(MIN_ROWS):0] num_rows_4x4;
    logic [$clog2(MIN_COLS):0] num_cols_4x4;
    logic vector_write_enable_4x4;
    logic [$clog2(MIN_COLS)-1:0] vector_base_addr_4x4;
    logic signed [DATA_WIDTH*BANDWIDTH_4X4-1:0] vector_in_4x4;
    logic [$clog2(MIN_ROWS*MIN_COLS)-1:0] matrix_addr_4x4;
    logic matrix_enable_4x4;
    logic signed [DATA_WIDTH*BANDWIDTH_4X4-1:0] matrix_data_4x4;
    logic matrix_ready_4x4;
    logic signed [(DATA_WIDTH*2)-1:0] result_out_4x4;
    logic result_valid_4x4;
    logic busy_4x4;

    // DUT 32x32 signals
    logic start_32x32;
    logic [$clog2(HALF_ROWS):0] num_rows_32x32;
    logic [$clog2(HALF_COLS):0] num_cols_32x32;
    logic vector_write_enable_32x32;
    logic [$clog2(HALF_COLS)-1:0] vector_base_addr_32x32;
    logic signed [DATA_WIDTH*MAX_BANDWIDTH-1:0] vector_in_32x32;
    logic [$clog2(HALF_ROWS*HALF_COLS)-1:0] matrix_addr_32x32;
    logic matrix_enable_32x32;
    logic signed [DATA_WIDTH*MAX_BANDWIDTH-1:0] matrix_data_32x32;
    logic matrix_ready_32x32;
    logic signed [(DATA_WIDTH*2)-1:0] result_out_32x32;
    logic result_valid_32x32;
    logic busy_32x32;

    // DUT 64x64 signals
    logic start_64x64;
    logic [$clog2(MAX_ROWS):0] num_rows_64x64;
    logic [$clog2(MAX_COLS):0] num_cols_64x64;
    logic vector_write_enable_64x64;
    logic [$clog2(MAX_COLS)-1:0] vector_base_addr_64x64;
    logic signed [DATA_WIDTH*MAX_BANDWIDTH-1:0] vector_in_64x64;
    logic [$clog2(MAX_ROWS*MAX_COLS)-1:0] matrix_addr_64x64;
    logic matrix_enable_64x64;
    logic signed [DATA_WIDTH*MAX_BANDWIDTH-1:0] matrix_data_64x64;
    logic matrix_ready_64x64;
    logic signed [(DATA_WIDTH*2)-1:0] result_out_64x64;
    logic result_valid_64x64;
    logic busy_64x64;

    //
    // === DUT instantiations ===
    //
    // For 4x4
    matvec_mult #(
        .MAX_ROWS(MIN_ROWS),
        .MAX_COLS(MIN_COLS),
        .BANDWIDTH(BANDWIDTH_4X4),
        .VEC_NUM_BANKS(VEC_NUM_BANKS_4X4)
    ) dut_4x4 (
        .clk(clk), .rst_n(rst_n), .start(start_4x4),
        .num_rows(num_rows_4x4), .num_cols(num_cols_4x4),
        .vector_write_enable(vector_write_enable_4x4),
        .vector_base_addr(vector_base_addr_4x4),
        .vector_in(vector_in_4x4),
        .matrix_addr(matrix_addr_4x4),
        .matrix_enable(matrix_enable_4x4),
        .matrix_data(matrix_data_4x4),
        .matrix_ready(matrix_ready_4x4),
        .result_out(result_out_4x4),
        .result_valid(result_valid_4x4),
        .busy(busy_4x4)
    );

    matrix_loader #(
        .NUM_ROWS(MIN_ROWS), .NUM_COLS(MIN_COLS), .DATA_WIDTH(DATA_WIDTH), .BANDWIDTH(BANDWIDTH_4X4)
    ) loader_4x4 (
        .clk(clk), .rst_n(rst_n),
        .enable(matrix_enable_4x4),
        .matrix_addr(matrix_addr_4x4),
        .matrix_data(matrix_data_4x4),
        .ready(matrix_ready_4x4)
    );

    // For 32x32
    matvec_mult #(
        .MAX_ROWS(HALF_ROWS),
        .MAX_COLS(HALF_COLS),
        .BANDWIDTH(MAX_BANDWIDTH),
        .VEC_NUM_BANKS(VEC_NUM_BANKS)
    ) dut_32x32 (
        .clk(clk), .rst_n(rst_n), .start(start_32x32),
        .num_rows(num_rows_32x32), .num_cols(num_cols_32x32),
        .vector_write_enable(vector_write_enable_32x32),
        .vector_base_addr(vector_base_addr_32x32),
        .vector_in(vector_in_32x32),
        .matrix_addr(matrix_addr_32x32),
        .matrix_enable(matrix_enable_32x32),
        .matrix_data(matrix_data_32x32),
        .matrix_ready(matrix_ready_32x32),
        .result_out(result_out_32x32),
        .result_valid(result_valid_32x32),
        .busy(busy_32x32)
    );

    matrix_loader #(
        .NUM_ROWS(HALF_ROWS), .NUM_COLS(HALF_COLS), .DATA_WIDTH(DATA_WIDTH), .BANDWIDTH(MAX_BANDWIDTH)
    ) loader_32x32 (
        .clk(clk), .rst_n(rst_n),
        .enable(matrix_enable_32x32),
        .matrix_addr(matrix_addr_32x32),
        .matrix_data(matrix_data_32x32),
        .ready(matrix_ready_32x32)
    );

    // For 64x64
    matvec_mult #(
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS),
        .BANDWIDTH(MAX_BANDWIDTH),
        .VEC_NUM_BANKS(VEC_NUM_BANKS)
    ) dut_64x64 (
        .clk(clk), .rst_n(rst_n), .start(start_64x64),
        .num_rows(num_rows_64x64), .num_cols(num_cols_64x64),
        .vector_write_enable(vector_write_enable_64x64),
        .vector_base_addr(vector_base_addr_64x64),
        .vector_in(vector_in_64x64),
        .matrix_addr(matrix_addr_64x64),
        .matrix_enable(matrix_enable_64x64),
        .matrix_data(matrix_data_64x64),
        .matrix_ready(matrix_ready_64x64),
        .result_out(result_out_64x64),
        .result_valid(result_valid_64x64),
        .busy(busy_64x64)
    );

    matrix_loader #(
        .NUM_ROWS(MAX_ROWS), .NUM_COLS(MAX_COLS), .DATA_WIDTH(DATA_WIDTH), .BANDWIDTH(MAX_BANDWIDTH)
    ) loader_64x64 (
        .clk(clk), .rst_n(rst_n),
        .enable(matrix_enable_64x64),
        .matrix_addr(matrix_addr_64x64),
        .matrix_data(matrix_data_64x64),
        .ready(matrix_ready_64x64)
    );

    //
    // === Test Tasks ===
    //
    task run_4x4_test();
        automatic int row = 0;
        automatic integer pass_count = 0;

        // Test vector (Q4.12 format)
        logic signed [DATA_WIDTH-1:0] test_vector [0:MIN_COLS-1];

        // Expected result (Q20.12 format)
        logic signed [(DATA_WIDTH*2)-1:0] expected_result [0:MIN_ROWS-1];

        // Define test vector: [1, 2, 3, 4] in Q4.12
        test_vector[0] = 16'sd4096;   // 1.0
        test_vector[1] = 16'sd8192;   // 2.0
        test_vector[2] = 16'sd12288;  // 3.0
        test_vector[3] = 16'sd16384;  // 4.0
        // Expected results
        expected_result[0] = 32'sd40960;   // 10.0 * 4096 = 40960
        expected_result[1] = 32'sd0;  // 0.0
        expected_result[2] = 32'sd40960;  // 10.0 * 4096 = 40960
        expected_result[3] = 32'sd0;  // 0.0

        num_rows_4x4 = MIN_ROWS;
        num_cols_4x4 = MIN_COLS;

        rst_n = 0; start_4x4 = 0; vector_write_enable_4x4 = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        start_4x4 = 1;
        @(posedge clk);
        start_4x4 = 0;

        vector_base_addr_4x4 = 0;
        for (int i = 0; i < BANDWIDTH_4X4; i++) begin
            vector_in_4x4[i*DATA_WIDTH +: DATA_WIDTH] = test_vector[i];
        end
        vector_write_enable_4x4 = 1;
        @(posedge clk);
        vector_write_enable_4x4 = 0;
        @(posedge clk);

        while (row < MIN_ROWS) begin
            @(posedge clk);
            if (result_valid_4x4) begin
                $display("[4x4] Row %0d: Got %0d, Expected %0d", row, result_out_4x4, expected_result[row]);
                if (result_out_4x4 === expected_result[row]) begin
                    $display("✅ PASS");
                    pass_count++;
                end else begin
                     $display("❌ FAIL");
                end
                row++;
            end
        end
        $display("[4x4] Test complete: %0d/4 passed", pass_count);
    endtask

    task run_32x32_test();
        automatic int row = 0;
        automatic integer pass_count = 0;
        automatic int expected;

        // Test vector (Q4.12 format)
        logic signed [DATA_WIDTH-1:0] test_vector [0:HALF_COLS-1];

        // Test vector (Q4.12 format)
        for (int i = 0; i < HALF_COLS; i++) test_vector[i] = 16'sd4096 * 3;

        num_rows_32x32 = HALF_ROWS;
        num_cols_32x32 = HALF_COLS;

        rst_n = 0; start_32x32 = 0; vector_write_enable_32x32 = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        start_32x32 = 1;
        @(posedge clk);
        start_32x32 = 0;

        for (int i = 0; i < HALF_COLS; i += MAX_BANDWIDTH) begin
            vector_base_addr_32x32 = i;
            for (int j = 0; j < MAX_BANDWIDTH; j++)
                vector_in_32x32[j*DATA_WIDTH +: DATA_WIDTH] = test_vector[i + j];
            vector_write_enable_32x32 = 1;
            @(posedge clk);
            vector_write_enable_32x32 = 0;
            @(posedge clk);
        end

        while (row < HALF_ROWS) begin
            @(posedge clk);
            if (result_valid_32x32) begin
                // Each row has half 1s and half 0s; Vector is all 3s
                // We should come up with a more varied set of matrix data at some point
                expected = 16 * 3 * 4096;
                $display("[32x32] Row %0d: Got %0d, Expected %0d", row, result_out_32x32, expected);
                if (result_out_32x32 === expected) begin
                    $display("✅ PASS");
                    pass_count++;
                end else begin
                    $display("❌ FAIL");
                end
                row++;
            end
        end
        $display("[32x32] Test complete: %0d/32 passed", pass_count);
    endtask

    task run_64x64_test();
        automatic int row = 0;
        automatic integer pass_count = 0;
        automatic int expected;

        // Test vector (Q4.12 format)
        logic signed [DATA_WIDTH-1:0] test_vector [0:MAX_COLS-1];

        // Test vector (Q4.12 format)
        for (int i = 0; i < MAX_COLS; i++) test_vector[i] = 16'sd4096;

        num_rows_64x64 = MAX_ROWS;
        num_cols_64x64 = MAX_COLS;

        rst_n = 0; start_64x64 = 0; vector_write_enable_64x64 = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        start_64x64 = 1;
        @(posedge clk);
        start_64x64 = 0;

        for (int i = 0; i < MAX_COLS; i += MAX_BANDWIDTH) begin
            vector_base_addr_64x64 = i;
            for (int j = 0; j < MAX_BANDWIDTH; j++)
                vector_in_64x64[j*DATA_WIDTH +: DATA_WIDTH] = test_vector[i + j];
            vector_write_enable_64x64 = 1;
            @(posedge clk);
            vector_write_enable_64x64 = 0;
            @(posedge clk);
        end

        while (row < MAX_ROWS) begin
            @(posedge clk);
            if (result_valid_64x64) begin
                // Each row has half 1s and half 0s
                // We should come up with a more varied set of matrix data at some point
                expected = 32 * 4096;
                $display("[64x64] Row %0d: Got %0d, Expected %0d", row, result_out_64x64, expected);
                if (result_out_64x64 === expected) begin
                    $display("✅ PASS");
                    pass_count++;
                end else begin
                    $display("❌ FAIL");
                end
                row++;
            end
        end
        $display("[64x64] Test complete: %0d/64 passed", pass_count);
    endtask

    initial begin
        run_4x4_test();
        repeat (5) @(posedge clk);
        run_32x32_test();
        repeat (5) @(posedge clk);
        run_64x64_test();
        $finish;
    end

endmodule
