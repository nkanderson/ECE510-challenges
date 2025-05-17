module matrix_loader #(
    // TODO: Will need to decide how many blocks of SRAM to use. This will determine
    // the address width and total size of each block.
    parameter ADDR_WIDTH = 12, // 2^12 = 4096
    parameter DATA_WIDTH = 16,
    parameter BANDWIDTH  = 16
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    input  logic [ADDR_WIDTH-1:0] matrix_addr, // Base address from multiplier
    output logic signed [DATA_WIDTH-1:0] matrix_data [0:BANDWIDTH-1],
    output logic ready
);

    // Matrix SRAM (placeholder), Q2.14 values
    logic signed [DATA_WIDTH-1:0] matrix_mem [0:(1<<ADDR_WIDTH)-1];

    // Internal state
    logic [ADDR_WIDTH-1:0] addr_reg;
    logic [$clog2(BANDWIDTH):0] chunk_offset;
    logic loading;

    // Initialize with placeholder ramp values
    initial begin
        for (int i = 0; i < (1<<ADDR_WIDTH); i++) begin
            matrix_mem[i] = $signed(i << 14);
        end
    end

    // FSM to fetch one matrix value per cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_reg <= '0;
            chunk_offset <= 0;
            loading <= 1'b0;
            ready <= 1'b0;
            for (int i = 0; i < BANDWIDTH; i++) begin
                matrix_data[i] <= '0;
            end
        end else begin
            if (enable && !loading) begin
                addr_reg <= matrix_addr;
                chunk_offset <= 0;
                loading <= 1'b1;
                ready <= 1'b0;
            end else if (loading) begin
                matrix_data[chunk_offset] <= matrix_mem[addr_reg + chunk_offset];
                chunk_offset <= chunk_offset + 1;

                if (chunk_offset == BANDWIDTH - 1) begin
                    loading <= 1'b0;
                    ready <= 1'b1;
                end else begin
                    ready <= 1'b0;
                end
            end else begin
                ready <= 1'b0;
            end
        end
    end

endmodule
