module matrix_loader (
	clk,
	rst_n,
	enable,
	matrix_addr,
	matrix_data,
	vccd1,
	vssd1,
	ready
);
	parameter signed [31:0] NUM_ROWS = 64;
	parameter signed [31:0] NUM_COLS = 64;
	parameter DATA_WIDTH = 16;
	parameter BANDWIDTH = 16;
	localparam signed [31:0] ADDR_WIDTH = $clog2(NUM_ROWS * NUM_COLS);
	input wire clk;
	input wire rst_n;
	input wire enable;
	input wire [ADDR_WIDTH - 1:0] matrix_addr;
	output reg [(DATA_WIDTH * BANDWIDTH) - 1:0] matrix_data;
	inout wire vccd1;
	inout wire vssd1;
	output reg ready;
	reg [ADDR_WIDTH - 1:0] addr_reg;
	reg [$clog2(BANDWIDTH):0] chunk_offset;
	reg loading;
	wire [31:0] sram_dout0;
	wire [31:0] sram_dout1;
	wire [31:0] sram_dout2;
	wire [31:0] sram_dout3;
	wire [31:0] unused_dout1;
	wire [8:0] sram_addr;
	wire [3:0] csb;
	wire [ADDR_WIDTH - 1:0] addr;
	wire [1:0] macro_index;
	wire halfword_select;
	reg [15:0] word;
	assign addr = addr_reg + {{ADDR_WIDTH - ($clog2(BANDWIDTH) >= 0 ? $clog2(BANDWIDTH) + 1 : 1 - $clog2(BANDWIDTH)) {1'b0}}, chunk_offset};
	assign macro_index = addr[11:10];
	assign halfword_select = addr[0];
	assign sram_addr = addr[9:1];
	assign csb = 4'b1111 & ~(4'b0001 << macro_index);
	always @(*)
		case (macro_index)
			2'd0: word = (halfword_select == 1'b0 ? sram_dout0[15:0] : sram_dout0[31:16]);
			2'd1: word = (halfword_select == 1'b0 ? sram_dout1[15:0] : sram_dout1[31:16]);
			2'd2: word = (halfword_select == 1'b0 ? sram_dout2[15:0] : sram_dout2[31:16]);
			2'd3: word = (halfword_select == 1'b0 ? sram_dout3[15:0] : sram_dout3[31:16]);
			default: word = 16'hxxxx;
		endcase
	sky130_sram_2kbyte_1rw1r_32x512_8 sram0(
		.clk0(clk),
		.csb0(csb[0]),
		.web0(1'b1),
		.wmask0(4'b0000),
		.addr0(sram_addr),
		.din0(32'b00000000000000000000000000000000),
		.dout0(sram_dout0),
		.clk1(1'b0),
		.csb1(1'b1),
		.addr1(9'b000000000),
		.dout1(unused_dout1),
		.vccd1(vccd1),
		.vssd1(vssd1)
	);
	sky130_sram_2kbyte_1rw1r_32x512_8 sram1(
		.clk0(clk),
		.csb0(csb[1]),
		.web0(1'b1),
		.wmask0(4'b0000),
		.addr0(sram_addr),
		.din0(32'b00000000000000000000000000000000),
		.dout0(sram_dout1),
		.clk1(1'b0),
		.csb1(1'b1),
		.addr1(9'b000000000),
		.dout1(unused_dout1),
		.vccd1(vccd1),
		.vssd1(vssd1)
	);
	sky130_sram_2kbyte_1rw1r_32x512_8 sram2(
		.clk0(clk),
		.csb0(csb[2]),
		.web0(1'b1),
		.wmask0(4'b0000),
		.addr0(sram_addr),
		.din0(32'b00000000000000000000000000000000),
		.dout0(sram_dout2),
		.clk1(1'b0),
		.csb1(1'b1),
		.addr1(9'b000000000),
		.dout1(unused_dout1),
		.vccd1(vccd1),
		.vssd1(vssd1)
	);
	sky130_sram_2kbyte_1rw1r_32x512_8 sram3(
		.clk0(clk),
		.csb0(csb[3]),
		.web0(1'b1),
		.wmask0(4'b0000),
		.addr0(sram_addr),
		.din0(32'b00000000000000000000000000000000),
		.dout0(sram_dout3),
		.clk1(1'b0),
		.csb1(1'b1),
		.addr1(9'b000000000),
		.dout1(unused_dout1),
		.vccd1(vccd1),
		.vssd1(vssd1)
	);
	always @(posedge clk)
		if (!rst_n) begin
			addr_reg <= {ADDR_WIDTH {1'b0}};
			chunk_offset <= 0;
			loading <= 1'b0;
			ready <= 1'b0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < BANDWIDTH; i = i + 1)
					matrix_data[i * DATA_WIDTH+:DATA_WIDTH] <= {DATA_WIDTH {1'b0}};
			end
		end
		else if ((enable && !loading) && !ready) begin
			addr_reg <= matrix_addr;
			chunk_offset <= 0;
			loading <= 1'b1;
			ready <= 1'b0;
		end
		else if (loading) begin
			matrix_data[chunk_offset * DATA_WIDTH+:DATA_WIDTH] <= $signed(word);
			chunk_offset <= chunk_offset + 1;
			if (chunk_offset == (BANDWIDTH - 1)) begin
				loading <= 1'b0;
				ready <= 1'b1;
			end
			else
				ready <= 1'b0;
		end
		else
			ready <= 1'b0;
endmodule
