module matrix_loader #(
    parameter ADDR_WIDTH = 12, // 2^12 = 4096
    parameter DATA_WIDTH = 16,
    parameter BANDWIDTH  = 16
)(
    input  logic clk,
    input  logic enable,
    input  logic [ADDR_WIDTH-1:0] matrix_addr, // Base address from multiplier
    output logic signed [DATA_WIDTH-1:0] matrix_data [0:BANDWIDTH-1],
    output logic ready
);

    // Matrix SRAM (placeholder for now), Q2.14 values
    logic signed [DATA_WIDTH-1:0] matrix_mem [0:(1<<ADDR_WIDTH)-1];

    // Initialize with placeholder ramp values
    initial begin
        for (int i = 0; i < (1<<ADDR_WIDTH); i++) begin
            matrix_mem[i] = $signed(i); // e.g., i = 5 -> 0.0003 in Q2.14
        end
    end

    // Output logic (combinational read)
    always_comb begin
        if (enable) begin
            for (int i = 0; i < BANDWIDTH; i++) begin
                matrix_data[i] = matrix_mem[matrix_addr + i];
            end
            ready = 1'b1;
        end else begin
            for (int i = 0; i < BANDWIDTH; i++) begin
                matrix_data[i] = '0;
            end
            ready = 1'b0;
        end
    end

endmodule
