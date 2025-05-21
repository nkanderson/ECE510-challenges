module mac4 (
	clk,
	a0,
	b0,
	a1,
	b1,
	a2,
	b2,
	a3,
	b3,
	result
);
	parameter DATA_WIDTH = 16;
	// NOTE: This circuit does not use a clock signal, but it is
	// required for use with OpenLane2.
	input clk;
	input wire signed [DATA_WIDTH - 1:0] a0;
	input wire signed [DATA_WIDTH - 1:0] b0;
	input wire signed [DATA_WIDTH - 1:0] a1;
	input wire signed [DATA_WIDTH - 1:0] b1;
	input wire signed [DATA_WIDTH - 1:0] a2;
	input wire signed [DATA_WIDTH - 1:0] b2;
	input wire signed [DATA_WIDTH - 1:0] a3;
	input wire signed [DATA_WIDTH - 1:0] b3;
	output wire signed [(DATA_WIDTH * 2) - 1:0] result;
	localparam RESULT_WIDTH = 2 * DATA_WIDTH;
	wire signed [RESULT_WIDTH - 1:0] p0;
	wire signed [RESULT_WIDTH - 1:0] p1;
	wire signed [RESULT_WIDTH - 1:0] p2;
	wire signed [RESULT_WIDTH - 1:0] p3;
	assign p0 = a0 * b0;
	assign p1 = a1 * b1;
	assign p2 = a2 * b2;
	assign p3 = a3 * b3;
	wire signed [RESULT_WIDTH - 1:0] s0;
	wire signed [RESULT_WIDTH - 1:0] s1;
	wire signed [RESULT_WIDTH - 1:0] s2;
	wire signed [RESULT_WIDTH - 1:0] s3;
	assign s0 = p0 >>> 14;
	assign s1 = p1 >>> 14;
	assign s2 = p2 >>> 14;
	assign s3 = p3 >>> 14;
	wire signed [RESULT_WIDTH - 1:0] sum;
	wire signed [RESULT_WIDTH - 1:0] adder_tree_1;
	wire signed [RESULT_WIDTH - 1:0] adder_tree_2;
	assign adder_tree_1 = s0 + s1;
	assign adder_tree_2 = s2 + s3;
	assign sum = adder_tree_1 + adder_tree_2;
	assign result = sum[RESULT_WIDTH - 1:0];
endmodule
