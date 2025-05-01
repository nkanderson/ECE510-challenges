`timescale 1ns / 1ps
`default_nettype none

module matvec_mul (
	clk,
	rst_n,
	start,
	done,
	matrix,
	vector,
	result
);
	reg _sv2v_0;
	parameter signed [31:0] VEC_SIZE = 6;
	parameter signed [31:0] OUT_SIZE = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire start;
	output wire done;
	input wire signed [((OUT_SIZE * VEC_SIZE) * DATA_WIDTH) - 1:0] matrix;
	input wire signed [(VEC_SIZE * DATA_WIDTH) - 1:0] vector;
	output reg signed [(OUT_SIZE * DATA_WIDTH) - 1:0] result;
	reg [2:0] state;
	reg [2:0] next_state;
	reg signed [DATA_WIDTH - 1:0] acc;
	reg signed [31:0] row_idx;
	reg signed [31:0] col_idx;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 3'd0;
		else
			state <= next_state;
	always @(*) begin
		if (_sv2v_0)
			;
		next_state = state;
		case (state)
			3'd0:
				if (start)
					next_state = 3'd1;
			3'd1: next_state = 3'd2;
			3'd2:
				if (col_idx == (VEC_SIZE - 1))
					next_state = 3'd3;
			3'd3:
				if (row_idx == (OUT_SIZE - 1))
					next_state = 3'd4;
				else
					next_state = 3'd1;
			3'd4:
				if (!start)
					next_state = 3'd0;
			default: next_state = 3'd0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			row_idx <= 0;
			col_idx <= 0;
			acc <= 0;
		end
		else
			case (state)
				3'd0: begin
					row_idx <= 0;
					col_idx <= 0;
					acc <= 0;
				end
				3'd1: begin
					col_idx <= 0;
					acc <= 0;
				end
				3'd2: begin
					acc <= acc + (matrix[((((OUT_SIZE - 1) - row_idx) * VEC_SIZE) + ((VEC_SIZE - 1) - col_idx)) * DATA_WIDTH+:DATA_WIDTH] * vector[((VEC_SIZE - 1) - col_idx) * DATA_WIDTH+:DATA_WIDTH]);
					if (col_idx < (VEC_SIZE - 1))
						col_idx <= col_idx + 1;
				end
				3'd3: begin
					result[((OUT_SIZE - 1) - row_idx) * DATA_WIDTH+:DATA_WIDTH] <= acc;
					row_idx <= row_idx + 1;
				end
				default:
					;
			endcase
	assign done = state == 3'd4;
	initial _sv2v_0 = 0;
endmodule
