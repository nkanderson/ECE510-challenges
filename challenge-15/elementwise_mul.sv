// -----------------------------------------------------------------------------
// elementwise_mul.sv - Elementwise multiplication module
// -----------------------------------------------------------------------------
module elementwise_mul #(
    parameter int VEC_SIZE = 32,
    parameter int DATA_WIDTH = 32
)(
    input  logic signed [DATA_WIDTH-1:0] a[VEC_SIZE],
    input  logic signed [DATA_WIDTH-1:0] b[VEC_SIZE],
    output logic signed [DATA_WIDTH-1:0] result[VEC_SIZE]
);

    genvar i;
    generate
        for (i = 0; i < VEC_SIZE; i++) begin : MUL_LOOP
            assign result[i] = a[i] * b[i];
        end
    endgenerate

endmodule
