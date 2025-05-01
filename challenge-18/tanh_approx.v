`timescale 1ns / 1ps
`default_nettype none

module tanh_approx (
	x,
	y
);
	reg _sv2v_0;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] FRAC_WIDTH = 16;
	input wire signed [DATA_WIDTH - 1:0] x;
	output reg signed [DATA_WIDTH - 1:0] y;
	reg signed [DATA_WIDTH - 1:0] x_abs;
	always @(*) begin
		if (_sv2v_0)
			;
		x_abs = (x < 0 ? -x : x);
		if (x >= 32'sd4096)
			y = (x >= 0 ? 32'sd65536 : -32'sd65536);
		else
			y = x;
	end
	initial _sv2v_0 = 0;
endmodule
