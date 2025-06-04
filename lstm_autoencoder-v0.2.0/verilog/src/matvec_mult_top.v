module matvec_mult_top #(
    parameter MAX_ROWS       = 64,
    parameter MAX_COLS       = 64,
    parameter DATA_WIDTH     = 16,
    parameter BANDWIDTH      = 16,
    parameter VEC_NUM_BANKS  = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic enable,
    input  logic vector_write_enable,
    input  logic [$clog2(MAX_COLS)-1:0] vector_base_addr,
    input  logic signed [(DATA_WIDTH * BANDWIDTH)-1:0] vector_in,
    input  logic [$clog2(MAX_ROWS):0] num_rows,
    input  logic [$clog2(MAX_COLS):0] num_cols,

    output logic signed [(DATA_WIDTH * 2)-1:0] result_out,
    output logic result_valid,
    output logic busy
);

    // Wires between matrix_loader and matvec_multiplier
    logic [$clog2(MAX_ROWS * MAX_COLS)-1:0] matrix_addr;
    logic matrix_enable;
    logic [(DATA_WIDTH * BANDWIDTH)-1:0] matrix_data;
    logic matrix_ready;
    wire vccd1 = 1'b1;
    wire vssd1 = 1'b0;

    // Instantiate the matrix loader
    matrix_loader #(
        .NUM_ROWS(MAX_ROWS),
        .NUM_COLS(MAX_COLS),
        .DATA_WIDTH(DATA_WIDTH),
        .BANDWIDTH(BANDWIDTH)
    ) matrix_loader_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable(matrix_enable),
        .matrix_addr(matrix_addr),
        .matrix_data(matrix_data),
        .ready(matrix_ready),
        .vccd1(vccd1),
        .vssd1(vssd1)
    );

    // Instantiate the matvec multiplier
    matvec_mult #(
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS),
        .DATA_WIDTH(DATA_WIDTH),
        .BANDWIDTH(BANDWIDTH),
        .VEC_NUM_BANKS(VEC_NUM_BANKS)
    ) matvec_mult_inst (
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

endmodule
