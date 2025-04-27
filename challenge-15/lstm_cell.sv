// -----------------------------------------------------------------------------
// lstm_cell.sv - LSTM Cell Hardware Implementation
// -----------------------------------------------------------------------------
module lstm_cell #(
    parameter int INPUT_SIZE  = 6,
    parameter int HIDDEN_SIZE = 32,
    parameter int DATA_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic signed [DATA_WIDTH-1:0] x[INPUT_SIZE],
    input  logic signed [DATA_WIDTH-1:0] h_prev[HIDDEN_SIZE],
    input  logic signed [DATA_WIDTH-1:0] c_prev[HIDDEN_SIZE],

    input  logic signed [DATA_WIDTH-1:0] W[HIDDEN_SIZE*4][INPUT_SIZE],
    input  logic signed [DATA_WIDTH-1:0] U[HIDDEN_SIZE*4][HIDDEN_SIZE],
    input  logic signed [DATA_WIDTH-1:0] b[HIDDEN_SIZE*4],

    output logic signed [DATA_WIDTH-1:0] h[HIDDEN_SIZE],
    output logic signed [DATA_WIDTH-1:0] c[HIDDEN_SIZE]
);

    // Internal wires for each gate
    logic signed [DATA_WIDTH-1:0] Wx_i[HIDDEN_SIZE], Wx_f[HIDDEN_SIZE], Wx_c[HIDDEN_SIZE], Wx_o[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] Uh_i[HIDDEN_SIZE], Uh_f[HIDDEN_SIZE], Uh_c[HIDDEN_SIZE], Uh_o[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] preact_i[HIDDEN_SIZE], preact_f[HIDDEN_SIZE], preact_c[HIDDEN_SIZE], preact_o[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] gate_i[HIDDEN_SIZE], gate_f[HIDDEN_SIZE], gate_c[HIDDEN_SIZE], gate_o[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] f_cprev[HIDDEN_SIZE], i_cand[HIDDEN_SIZE], c_tanh[HIDDEN_SIZE];

    // Matrix-vector multiplies for Wx and Uh per gate
    matvec_mul #(INPUT_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Wx_i (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(W[0:HIDDEN_SIZE-1]), .vector(x), .result(Wx_i));
    matvec_mul #(HIDDEN_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Uh_i (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(U[0:HIDDEN_SIZE-1]), .vector(h_prev), .result(Uh_i));

    matvec_mul #(INPUT_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Wx_f (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(W[HIDDEN_SIZE:HIDDEN_SIZE*2-1]), .vector(x), .result(Wx_f));
    matvec_mul #(HIDDEN_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Uh_f (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(U[HIDDEN_SIZE:HIDDEN_SIZE*2-1]), .vector(h_prev), .result(Uh_f));

    matvec_mul #(INPUT_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Wx_c (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(W[HIDDEN_SIZE*2:HIDDEN_SIZE*3-1]), .vector(x), .result(Wx_c));
    matvec_mul #(HIDDEN_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Uh_c (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(U[HIDDEN_SIZE*2:HIDDEN_SIZE*3-1]), .vector(h_prev), .result(Uh_c));

    matvec_mul #(INPUT_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Wx_o (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(W[HIDDEN_SIZE*3:HIDDEN_SIZE*4-1]), .vector(x), .result(Wx_o));
    matvec_mul #(HIDDEN_SIZE, HIDDEN_SIZE, DATA_WIDTH) matvec_Uh_o (.clk(clk), .rst_n(rst_n), .start(start), .done(), .matrix(U[HIDDEN_SIZE*3:HIDDEN_SIZE*4-1]), .vector(h_prev), .result(Uh_o));

    // Elementwise additions for Wx + Uh + b per gate
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_Wx_Uh_i (.a(Wx_i), .b(Uh_i), .result(preact_i));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_Wx_Uh_f (.a(Wx_f), .b(Uh_f), .result(preact_f));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_Wx_Uh_c (.a(Wx_c), .b(Uh_c), .result(preact_c));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_Wx_Uh_o (.a(Wx_o), .b(Uh_o), .result(preact_o));

    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_bias_i (.a(preact_i), .b(b[0:HIDDEN_SIZE-1]), .result(preact_i));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_bias_f (.a(preact_f), .b(b[HIDDEN_SIZE:HIDDEN_SIZE*2-1]), .result(preact_f));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_bias_c (.a(preact_c), .b(b[HIDDEN_SIZE*2:HIDDEN_SIZE*3-1]), .result(preact_c));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_bias_o (.a(preact_o), .b(b[HIDDEN_SIZE*3:HIDDEN_SIZE*4-1]), .result(preact_o));

    // Activation functions
    generate
        genvar idx;
        for (idx = 0; idx < HIDDEN_SIZE; idx++) begin
            sigmoid_approx #(DATA_WIDTH) sigmoid_i_inst (.x(preact_i[idx]), .y(gate_i[idx]));
            sigmoid_approx #(DATA_WIDTH) sigmoid_f_inst (.x(preact_f[idx]), .y(gate_f[idx]));
            tanh_approx #(DATA_WIDTH) tanh_c_inst (.x(preact_c[idx]), .y(gate_c[idx]));
            sigmoid_approx #(DATA_WIDTH) sigmoid_o_inst (.x(preact_o[idx]), .y(gate_o[idx]));
        end
    endgenerate

    // Cell state and hidden state calculations
    elementwise_mul #(HIDDEN_SIZE, DATA_WIDTH) mul_f_cprev (.a(gate_f), .b(c_prev), .result(f_cprev));
    elementwise_mul #(HIDDEN_SIZE, DATA_WIDTH) mul_i_cand (.a(gate_i), .b(gate_c), .result(i_cand));
    elementwise_add #(HIDDEN_SIZE, DATA_WIDTH) add_cell (.a(f_cprev), .b(i_cand), .result(c));

    generate
        for (idx = 0; idx < HIDDEN_SIZE; idx++) begin
            tanh_approx #(DATA_WIDTH) tanh_cout (.x(c[idx]), .y(c_tanh[idx]));
        end
    endgenerate

    elementwise_mul #(HIDDEN_SIZE, DATA_WIDTH) mul_o_ctanh (.a(gate_o), .b(c_tanh), .result(h));

    // Done signal (simple - combinational completion)
    assign done = 1'b1;

endmodule
// -----------------------------------------------------------------------------
