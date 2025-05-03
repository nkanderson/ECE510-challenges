module tb_systolic_sorter;
	reg _sv2v_0;
	parameter N = 8;
	reg clk = 0;
	reg rst = 1;
	reg load = 0;
	reg [(N * 32) - 1:0] in_data_flat;
	wire [(N * 32) - 1:0] out_data_flat;
	systolic_sorter #(.N(N)) dut(
		.clk(clk),
		.rst(rst),
		.load(load),
		.in_data_flat(in_data_flat),
		.out_data_flat(out_data_flat)
	);
	always #(5) clk = ~clk;
	task apply_input;
		input reg [(N * 32) - 1:0] data;
		reg signed [31:0] i;
		for (i = 0; i < N; i = i + 1)
			in_data_flat[i * 32+:32] = data[((N - 1) - i) * 32+:32];
	endtask
	function check_sorted;
		input reg _sv2v_unused;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				begin : sv2v_autoblock_2
					reg signed [31:0] _sv2v_value_on_break;
					for (i = 1; i < N; i = i + 1)
						if (_sv2v_jump < 2'b10) begin
							_sv2v_jump = 2'b00;
							if (out_data_flat[(i - 1) * 32+:32] > out_data_flat[i * 32+:32]) begin
								check_sorted = 0;
								_sv2v_jump = 2'b11;
							end
							_sv2v_value_on_break = i;
						end
					if (!(_sv2v_jump < 2'b10))
						i = _sv2v_value_on_break;
					if (_sv2v_jump != 2'b11)
						_sv2v_jump = 2'b00;
				end
			end
			if (_sv2v_jump == 2'b00) begin
				check_sorted = 1;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	task run_test;
		input string label;
		input reg [(N * 32) - 1:0] test_data;
		begin
			$display("Running test: %s", label);
			apply_input(test_data);
			rst = 1;
			load = 0;
			#(10)
				;
			rst = 0;
			load = 1;
			#(10)
				;
			load = 0;
			repeat (N * 2) @(posedge clk)
				;
			if (check_sorted(0))
				$display("PASS: %s", label);
			else begin
				$display("FAIL: %s", label);
				begin : sv2v_autoblock_3
					reg signed [31:0] i;
					for (i = 0; i < N; i = i + 1)
						$display("[%0d] = %0d", i, out_data_flat[i * 32+:32]);
				end
			end
		end
	endtask
	initial begin : sv2v_autoblock_4
		reg [(N * 32) - 1:0] test1;
		reg [(N * 32) - 1:0] test2;
		reg [(N * 32) - 1:0] test3;
		test1 = 256'h0000000800000007000000060000000500000004000000030000000200000001;
		test2 = 256'h0000000100000003000000050000000700000002000000040000000600000008;
		test3 = 256'h0000000900000002000000060000000300000007000000050000000100000004;
		run_test("Descending Order", test1);
		run_test("Alternating High-Low", test2);
		run_test("Random Mix", test3);
		$finish;
	end
	initial _sv2v_0 = 0;
endmodule
