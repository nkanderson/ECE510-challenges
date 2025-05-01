`timescale 1ns / 1ps
`default_nettype none

module elementwise_add (
	a,
	b,
	result
);
	parameter signed [31:0] VEC_SIZE = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	input wire signed [(VEC_SIZE * DATA_WIDTH) - 1:0] a;
	input wire signed [(VEC_SIZE * DATA_WIDTH) - 1:0] b;
	output wire signed [(VEC_SIZE * DATA_WIDTH) - 1:0] result;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < VEC_SIZE; _gv_i_1 = _gv_i_1 + 1) begin : ADD_LOOP
			localparam i = _gv_i_1;
			assign result[((VEC_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH] = a[((VEC_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH] + b[((VEC_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH];
		end
	endgenerate
endmodule
