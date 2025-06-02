module matrix_loader #(
    parameter int NUM_ROWS = 64,
    parameter int NUM_COLS = 64,
    parameter DATA_WIDTH = 16,
    parameter BANDWIDTH  = 16,
    localparam int ADDR_WIDTH = $clog2(NUM_ROWS * NUM_COLS)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    // TODO: May want to base ADDR_WIDTH on set rows and columns from matvec_multiplier
    // rather than a specific address width
    input  logic [ADDR_WIDTH-1:0] matrix_addr, // Base address from multiplier
    output logic [DATA_WIDTH*BANDWIDTH-1:0] matrix_data,
    output logic ready
);

    // Internal state
    logic [ADDR_WIDTH-1:0] addr_reg;
    logic [$clog2(BANDWIDTH):0] chunk_offset;
    logic loading;

`ifdef USE_SRAM_MACRO
    // Flattened 32-bit outputs from 4 SRAM macros
    logic [31:0] sram_dout0, sram_dout1, sram_dout2, sram_dout3;
    wire [31:0] unused_dout1;
    logic [8:0] sram_addr;
    logic [3:0] csb;
    logic [ADDR_WIDTH-1:0] addr;
    logic [1:0] macro_index;
    logic halfword_select;
    logic [15:0] word;

    assign addr = addr_reg + chunk_offset;
    assign macro_index = addr[11:10];
    assign halfword_select = addr[0];
    assign sram_addr = addr[9:1];
    assign csb = 4'b1111 & ~(1'b1 << macro_index);

    always_comb begin
        case (macro_index)
            2'd0: word = (halfword_select == 1'b0) ? sram_dout0[15:0]  : sram_dout0[31:16];
            2'd1: word = (halfword_select == 1'b0) ? sram_dout1[15:0]  : sram_dout1[31:16];
            2'd2: word = (halfword_select == 1'b0) ? sram_dout2[15:0]  : sram_dout2[31:16];
            2'd3: word = (halfword_select == 1'b0) ? sram_dout3[15:0]  : sram_dout3[31:16];
            default: word = 16'hxxxx;
        endcase
    end

    // SRAM macros (sky130_sram_2kbyte_1rw1r_32x512_8)
    // https://github.com/VLSIDA/sky130_sram_macros/blob/main/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8.v
    sky130_sram_2kbyte_1rw1r_32x512_8 sram0 (
        .clk0(clk), .csb0(csb[0]), .web0(1'b1), .wmask0(4'b0000),
        .addr0(sram_addr), .din0(32'b0), .dout0(sram_dout0),
        .clk1(1'b0), .csb1(1'b1), .addr1(9'b0), .dout1(unused_dout1)
    );
    sky130_sram_2kbyte_1rw1r_32x512_8 sram1 (
        .clk0(clk), .csb0(csb[1]), .web0(1'b1), .wmask0(4'b0000),
        .addr0(sram_addr), .din0(32'b0), .dout0(sram_dout1),
        .clk1(1'b0), .csb1(1'b1), .addr1(9'b0), .dout1(unused_dout1)
    );
    sky130_sram_2kbyte_1rw1r_32x512_8 sram2 (
        .clk0(clk), .csb0(csb[2]), .web0(1'b1), .wmask0(4'b0000),
        .addr0(sram_addr), .din0(32'b0), .dout0(sram_dout2),
        .clk1(1'b0), .csb1(1'b1), .addr1(9'b0), .dout1(unused_dout1)
    );
    sky130_sram_2kbyte_1rw1r_32x512_8 sram3 (
        .clk0(clk), .csb0(csb[3]), .web0(1'b1), .wmask0(4'b0000),
        .addr0(sram_addr), .din0(32'b0), .dout0(sram_dout3),
        .clk1(1'b0), .csb1(1'b1), .addr1(9'b0), .dout1(unused_dout1)
    );
`else
    // Behavioral memory for simulation
    logic signed [DATA_WIDTH-1:0] matrix_mem [0:(1<<ADDR_WIDTH)-1];
    // TODO: Generate golden vector output from Python inference script to
    // determine expected output using this weight. Also, confirm filepath usage here.
    // initial $readmemh("../data/enc1_U_i.mem", matrix_mem);
    // Until we have golden vectors, initialize with placeholder values
    initial begin
        for (int i = 0; i < (1<<ADDR_WIDTH); i++) begin
            // For now, load alternating BANDWIDTH-sized chunks of 1s and 0s
            // If we want to do full rows, we need to know the number of columns
            matrix_mem[i] = (i % (BANDWIDTH * 2) < BANDWIDTH) ? $signed(16'(1'b1 << 14)) : '0;
        end
    end
`endif

    // FSM to fetch one matrix value per cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_reg <= '0;
            chunk_offset <= 0;
            loading <= 1'b0;
            ready <= 1'b0;
            for (int i = 0; i < BANDWIDTH; i++) begin
                matrix_data[i*DATA_WIDTH +: DATA_WIDTH] <= '0;
            end
        end else begin
            if (enable && !loading &&!ready) begin
                addr_reg <= matrix_addr;
                chunk_offset <= 0;
                loading <= 1'b1;
                ready <= 1'b0;
            end else if (loading) begin
`ifdef USE_SRAM_MACRO
                matrix_data[chunk_offset*DATA_WIDTH +: DATA_WIDTH] <= $signed(word);
`else
                matrix_data[chunk_offset*DATA_WIDTH +: DATA_WIDTH] <= matrix_mem[addr_reg + chunk_offset];
`endif
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
