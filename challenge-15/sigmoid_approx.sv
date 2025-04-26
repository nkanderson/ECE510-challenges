// -----------------------------------------------------------------------------
// sigmoid_approx.sv - Piecewise linear sigmoid approximation
// -----------------------------------------------------------------------------
module sigmoid_approx #(
    parameter int DATA_WIDTH = 32,
    parameter int FRAC_WIDTH = 16  // For fixed-point scaling
)(
    input  logic signed [DATA_WIDTH-1:0] x,
    output logic signed [DATA_WIDTH-1:0] y
);

    // Fixed-point inputs assumed
    logic signed [DATA_WIDTH-1:0] x_abs;

    always_comb begin
        x_abs = (x < 0) ? -x : x;

        if (x >= 32'sd4096) begin
            y = 32'sd65536; // 1.0 in Q16 format
        end else if (x <= -32'sd4096) begin
            y = 32'sd0; // 0.0 in Q16 format
        end else begin
            // Linear segments for x in [-4096, 4096] (~[-0.0625, 0.0625] in real)
            y = (x >>> 2) + 32'sd32768; // y = 0.25x + 0.5
        end
    end

endmodule
