`timescale 1ns / 1ps
`default_nettype none

module systolic_sorter (
	clk,
	rst,
	load,
	in_data_flat,
	out_data_flat
);
	reg _sv2v_0;
	parameter N = 8;
	input wire clk;
	input wire rst;
	input wire load;
	input wire [(N * 32) - 1:0] in_data_flat;
	output reg [(N * 32) - 1:0] out_data_flat;
	reg [31:0] pe_inputs [0:N - 1];
	reg [31:0] unpacked_input [0:N - 1];
	reg [31:0] pe_outputs [0:N - 1];
	wire [31:0] gen_left [0:N - 2];
	wire [31:0] gen_right [0:N - 2];
	reg phase;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < N; i = i + 1)
				unpacked_input[i] = in_data_flat[i * 32+:32];
		end
	end
	always @(posedge clk or posedge rst)
		if (rst) begin
			begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < N; i = i + 1)
					pe_inputs[i] <= 0;
			end
			phase <= 0;
		end
		else if (load) begin
			begin : sv2v_autoblock_3
				reg signed [31:0] i;
				for (i = 0; i < N; i = i + 1)
					pe_inputs[i] <= unpacked_input[i];
			end
			phase <= 0;
		end
		else begin
			begin : sv2v_autoblock_4
				reg signed [31:0] i;
				for (i = 0; i < N; i = i + 1)
					pe_inputs[i] <= pe_outputs[i];
			end
			phase <= ~phase;
		end
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < (N - 1); _gv_i_1 = _gv_i_1 + 1) begin : gen_pe
			localparam i = _gv_i_1;
			pe pe_inst(
				.clk(clk),
				.rst(rst),
				.in_left(pe_inputs[i]),
				.in_right(pe_inputs[i + 1]),
				.sel(((phase == 0) && ((i % 2) == 0)) || ((phase == 1) && ((i % 2) == 1))),
				.out_left(gen_left[i]),
				.out_right(gen_right[i])
			);
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_5
			reg signed [31:0] i;
			for (i = 0; i < N; i = i + 1)
				pe_outputs[i] = pe_inputs[i];
		end
		begin : sv2v_autoblock_6
			reg signed [31:0] i;
			for (i = 0; i < (N - 1); i = i + 1)
				if (((phase == 0) && ((i % 2) == 0)) || ((phase == 1) && ((i % 2) == 1))) begin
					pe_outputs[i] = gen_left[i];
					pe_outputs[i + 1] = gen_right[i];
				end
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_7
			reg signed [31:0] i;
			for (i = 0; i < N; i = i + 1)
				out_data_flat[i * 32+:32] = pe_inputs[i];
		end
	end
	initial _sv2v_0 = 0;
endmodule
