module matvec_multiplier (
	clk,
	rst_n,
	start,
	num_rows,
	num_cols,
	vector_write_enable,
	vector_base_addr,
	vector_in,
	matrix_addr,
	matrix_enable,
	matrix_data,
	matrix_ready,
	result_out,
	result_valid,
	busy
);
	parameter MAX_ROWS = 64;
	parameter MAX_COLS = 64;
	parameter BANDWIDTH = 16;
	parameter DATA_WIDTH = 16;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [$clog2(MAX_ROWS):0] num_rows;
	input wire [$clog2(MAX_COLS):0] num_cols;
	input wire vector_write_enable;
	input wire [$clog2(MAX_COLS) - 1:0] vector_base_addr;
	input wire signed [(DATA_WIDTH * BANDWIDTH) - 1:0] vector_in;
	output wire [$clog2(MAX_ROWS * MAX_COLS) - 1:0] matrix_addr;
	output wire matrix_enable;
	input wire [(DATA_WIDTH * BANDWIDTH) - 1:0] matrix_data;
	input wire matrix_ready;
	output reg signed [(DATA_WIDTH * 2) - 1:0] result_out;
	output reg result_valid;
	output wire busy;
	reg [5:0] state;
	reg [5:0] next_state;
	reg [$clog2(MAX_ROWS) - 1:0] row_idx;
	reg [$clog2(MAX_COLS) - 1:0] col_idx;
	localparam MAC_CHUNK_SIZE = 4;
	reg [BANDWIDTH - 1:0] num_ops;
	reg signed [DATA_WIDTH * 2:0] acc;
	reg signed [(DATA_WIDTH * MAX_COLS) - 1:0] vector_buffer;
	reg vector_loaded;
	wire signed [(DATA_WIDTH * 2) - 1:0] mac_result;
	wire [(DATA_WIDTH * MAC_CHUNK_SIZE) - 1:0] a;
	wire [(DATA_WIDTH * MAC_CHUNK_SIZE) - 1:0] b;
	mac4 mac(
		.a0(a[0+:DATA_WIDTH]),
		.b0(b[0+:DATA_WIDTH]),
		.a1(a[1 * DATA_WIDTH+:DATA_WIDTH]),
		.b1(b[1 * DATA_WIDTH+:DATA_WIDTH]),
		.a2(a[2 * DATA_WIDTH+:DATA_WIDTH]),
		.b2(b[2 * DATA_WIDTH+:DATA_WIDTH]),
		.a3(a[3 * DATA_WIDTH+:DATA_WIDTH]),
		.b3(b[3 * DATA_WIDTH+:DATA_WIDTH]),
		.result(mac_result)
	);
	assign busy = state != 6'b000001;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 6'b000001;
		else
			state <= next_state;
	always @(*) begin
		next_state = state;
		(* full_case, parallel_case *)
		case (state)
			6'b000001:
				if (start)
					next_state = 6'b000010;
			6'b000010:
				if (vector_loaded)
					next_state = 6'b000100;
			6'b000100: next_state = 6'b001000;
			6'b001000:
				if (matrix_ready)
					next_state = 6'b010000;
			6'b010000:
				if ((num_ops + MAC_CHUNK_SIZE) >= BANDWIDTH)
					next_state = (row_idx < num_rows ? 6'b000100 : 6'b100000);
				else
					next_state = 6'b010000;
			6'b100000: next_state = 6'b000001;
			default: next_state = state;
		endcase
	end
	always @(posedge clk or negedge rst_n) begin : sv2v_autoblock_1
		integer i;
		if (!rst_n)
			for (i = 0; i < MAX_COLS; i = i + 1)
				vector_buffer[i] <= 0;
		else if (vector_write_enable)
			for (i = 0; i < BANDWIDTH; i = i + 1)
				vector_buffer[(vector_base_addr + i) * DATA_WIDTH+:DATA_WIDTH] <= vector_in[i * DATA_WIDTH+:DATA_WIDTH];
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			vector_loaded <= 0;
		else if (vector_write_enable) begin
			if ((vector_base_addr + BANDWIDTH) >= num_cols)
				vector_loaded <= 1;
			else
				vector_loaded <= 0;
		end
		else if (state == 6'b000001)
			vector_loaded <= 0;
	assign matrix_enable = (state == 6'b000100) || (state == 6'b001000);
	assign matrix_addr = (row_idx * num_cols) + col_idx;
	reg [$clog2(MAX_ROWS) - 1:0] row_idx_next;
	reg [$clog2(MAX_COLS) - 1:0] col_idx_next;
	always @(*) begin
		row_idx_next = row_idx;
		col_idx_next = col_idx;
		if (state == 6'b010000) begin
			if (((col_idx + MAC_CHUNK_SIZE) >= num_cols) && (row_idx < (num_rows - 1))) begin
				row_idx_next = row_idx + 1;
				col_idx_next = 0;
			end
			else
				col_idx_next = col_idx + MAC_CHUNK_SIZE;
		end
		else if (state == 6'b000001) begin
			col_idx_next = 0;
			row_idx_next = 0;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			row_idx <= 0;
			col_idx <= 0;
		end
		else begin
			row_idx <= row_idx_next;
			col_idx <= col_idx_next;
		end
	reg [BANDWIDTH - 1:0] num_ops_next;
	always @(*)
		if (state == 6'b010000)
			num_ops_next = num_ops + MAC_CHUNK_SIZE;
		else if ((state == 6'b001000) || (state == 6'b000001))
			num_ops_next = 0;
		else
			num_ops_next = num_ops;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			num_ops <= 0;
		else
			num_ops <= num_ops_next;
	reg signed [DATA_WIDTH * 2:0] acc_next;
	always @(*)
		if ((col_idx + MAC_CHUNK_SIZE) >= num_cols)
			acc_next = 0;
		else if (state == 6'b010000)
			acc_next = acc + mac_result;
		else
			acc_next = acc;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			acc <= 0;
		else
			acc <= acc_next;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < MAC_CHUNK_SIZE; _gv_i_1 = _gv_i_1 + 1) begin : mac_inputs
			localparam i = _gv_i_1;
			assign a[i * DATA_WIDTH+:DATA_WIDTH] = ((state == 6'b010000) && ((col_idx + i) < num_cols) ? matrix_data[((col_idx % BANDWIDTH) + i) * DATA_WIDTH+:DATA_WIDTH] : {DATA_WIDTH * 1 {1'sb0}});
			assign b[i * DATA_WIDTH+:DATA_WIDTH] = ((state == 6'b010000) && ((col_idx + i) < num_cols) ? vector_buffer[(col_idx + i) * DATA_WIDTH+:DATA_WIDTH] : {DATA_WIDTH * 1 {1'sb0}});
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			result_out <= 0;
			result_valid <= 0;
		end
		else if ((state == 6'b010000) && ((col_idx + MAC_CHUNK_SIZE) >= num_cols)) begin
			result_out <= acc + mac_result;
			result_valid <= 1;
		end
		else
			result_valid <= 0;
endmodule
