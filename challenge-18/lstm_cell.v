`timescale 1ns / 1ps
`default_nettype none

module lstm_cell (
	clk,
	rst_n,
	start,
	done,
	x,
	h_prev,
	c_prev,
	W,
	U,
	b,
	h,
	c
);
	reg _sv2v_0;
	parameter signed [31:0] INPUT_SIZE = 2;
	parameter signed [31:0] HIDDEN_SIZE = 2;
	parameter signed [31:0] DATA_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire start;
	output wire done;
	input wire signed [(INPUT_SIZE * DATA_WIDTH) - 1:0] x;
	input wire signed [(HIDDEN_SIZE * DATA_WIDTH) - 1:0] h_prev;
	input wire signed [(HIDDEN_SIZE * DATA_WIDTH) - 1:0] c_prev;
	input wire signed [(((HIDDEN_SIZE * 4) * INPUT_SIZE) * DATA_WIDTH) - 1:0] W;
	input wire signed [(((HIDDEN_SIZE * 4) * HIDDEN_SIZE) * DATA_WIDTH) - 1:0] U;
	input wire signed [((HIDDEN_SIZE * 4) * DATA_WIDTH) - 1:0] b;
	output reg signed [(HIDDEN_SIZE * DATA_WIDTH) - 1:0] h;
	output reg signed [(HIDDEN_SIZE * DATA_WIDTH) - 1:0] c;
	reg [6:0] state;
	reg [6:0] next_state;
	reg start_Wx;
	wire done_Wx;
	reg start_Uh;
	wire done_Uh;
	wire signed [((HIDDEN_SIZE * 4) * DATA_WIDTH) - 1:0] Wx_total;
	wire signed [((HIDDEN_SIZE * 4) * DATA_WIDTH) - 1:0] Uh_total;
	reg signed [DATA_WIDTH - 1:0] preact [0:(HIDDEN_SIZE * 4) - 1];
	wire signed [DATA_WIDTH - 1:0] gate_i [0:HIDDEN_SIZE - 1];
	wire signed [DATA_WIDTH - 1:0] gate_f [0:HIDDEN_SIZE - 1];
	wire signed [DATA_WIDTH - 1:0] gate_c [0:HIDDEN_SIZE - 1];
	wire signed [DATA_WIDTH - 1:0] gate_o [0:HIDDEN_SIZE - 1];
	wire signed [DATA_WIDTH - 1:0] tanh_c [0:HIDDEN_SIZE - 1];
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 7'b0000001;
		else
			state <= next_state;
	always @(*) begin
		if (_sv2v_0)
			;
		next_state = state;
		start_Wx = 1'b0;
		start_Uh = 1'b0;
		case (state)
			7'b0000001:
				if (start)
					next_state = 7'b0000010;
			7'b0000010: begin
				start_Wx = 1'b1;
				if (done_Wx)
					next_state = 7'b0000100;
			end
			7'b0000100: begin
				start_Uh = 1'b1;
				if (done_Uh)
					next_state = 7'b0001000;
			end
			7'b0001000: next_state = 7'b0010000;
			7'b0010000: next_state = 7'b0100000;
			7'b0100000: next_state = 7'b1000000;
			7'b1000000: next_state = 7'b0000001;
			default: next_state = 7'b0000001;
		endcase
	end
	assign done = state == 7'b1000000;
	matvec_mul #(
		.VEC_SIZE(INPUT_SIZE),
		.OUT_SIZE(HIDDEN_SIZE * 4),
		.DATA_WIDTH(DATA_WIDTH)
	) u_matvec_Wx(
		.clk(clk),
		.rst_n(rst_n),
		.start(start_Wx),
		.done(done_Wx),
		.matrix(W),
		.vector(x),
		.result(Wx_total)
	);
	matvec_mul #(
		.VEC_SIZE(HIDDEN_SIZE),
		.OUT_SIZE(HIDDEN_SIZE * 4),
		.DATA_WIDTH(DATA_WIDTH)
	) u_matvec_Uh(
		.clk(clk),
		.rst_n(rst_n),
		.start(start_Uh),
		.done(done_Uh),
		.matrix(U),
		.vector(h_prev),
		.result(Uh_total)
	);
	always @(*) begin : sv2v_autoblock_1
		integer i;
		if (_sv2v_0)
			;
		for (i = 0; i < (HIDDEN_SIZE * 4); i = i + 1)
			preact[i] = (Wx_total[(((HIDDEN_SIZE * 4) - 1) - i) * DATA_WIDTH+:DATA_WIDTH] + Uh_total[(((HIDDEN_SIZE * 4) - 1) - i) * DATA_WIDTH+:DATA_WIDTH]) + b[(((HIDDEN_SIZE * 4) - 1) - i) * DATA_WIDTH+:DATA_WIDTH];
	end
	genvar _gv_idx_1;
	generate
		for (_gv_idx_1 = 0; _gv_idx_1 < HIDDEN_SIZE; _gv_idx_1 = _gv_idx_1 + 1) begin : genblk1
			localparam idx = _gv_idx_1;
			sigmoid_approx u_sigmoid_i(
				.x(preact[idx]),
				.y(gate_i[idx])
			);
			sigmoid_approx u_sigmoid_f(
				.x(preact[HIDDEN_SIZE + idx]),
				.y(gate_f[idx])
			);
			tanh_approx u_tanh_c(
				.x(preact[(HIDDEN_SIZE * 2) + idx]),
				.y(gate_c[idx])
			);
			sigmoid_approx u_sigmoid_o(
				.x(preact[(HIDDEN_SIZE * 3) + idx]),
				.y(gate_o[idx])
			);
			tanh_approx u_tanh_c2(
				.x(c[((HIDDEN_SIZE - 1) - idx) * DATA_WIDTH+:DATA_WIDTH]),
				.y(tanh_c[idx])
			);
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			begin : sv2v_autoblock_2
				integer i;
				for (i = 0; i <= (HIDDEN_SIZE - 1); i = i + 1)
					h[((HIDDEN_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH] <= 0;
			end
			begin : sv2v_autoblock_3
				integer i;
				for (i = 0; i <= (HIDDEN_SIZE - 1); i = i + 1)
					c[((HIDDEN_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH] <= 0;
			end
		end
		else if (state == 7'b0100000) begin : sv2v_autoblock_4
			integer i;
			for (i = 0; i < HIDDEN_SIZE; i = i + 1)
				begin
					c[((HIDDEN_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH] <= (gate_f[i] * c_prev[((HIDDEN_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH]) + (gate_i[i] * gate_c[i]);
					h[((HIDDEN_SIZE - 1) - i) * DATA_WIDTH+:DATA_WIDTH] <= gate_o[i] * tanh_c[i];
				end
		end
	initial _sv2v_0 = 0;
endmodule
