`timescale 1ns / 1ps
`default_nettype none

module pe (
	clk,
	rst,
	in_left,
	in_right,
	sel,
	out_left,
	out_right
);
	input wire clk;
	input wire rst;
	input wire [31:0] in_left;
	input wire [31:0] in_right;
	input wire sel;
	output reg [31:0] out_left;
	output reg [31:0] out_right;
	always @(posedge clk or posedge rst)
		if (rst) begin
			out_left <= 0;
			out_right <= 0;
		end
		else if (sel) begin
			if (in_left > in_right) begin
				out_left <= in_right;
				out_right <= in_left;
			end
			else begin
				out_left <= in_left;
				out_right <= in_right;
			end
		end
		else begin
			out_left <= in_left;
			out_right <= in_right;
		end
endmodule
