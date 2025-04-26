// -----------------------------------------------------------------------------
// tanh_approx.sv - Piecewise linear tanh approximation
// -----------------------------------------------------------------------------
module tanh_approx #(
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
            y = (x >= 0) ? 32'sd65536 : -32'sd65536; // +1 or -1 in Q16
        end else begin
            // Linear region for small x
            y = x; // Approximate tanh(x) ~ x for small x
        end
    end

endmodule
