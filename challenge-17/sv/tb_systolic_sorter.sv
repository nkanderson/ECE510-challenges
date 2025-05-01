// tb_systolic_sorter.sv (updated with multiple test cases and self-checking)
module tb_systolic_sorter;
    parameter N = 8;

    logic clk = 0;
    logic rst = 1;
    logic load = 0;
    logic [N*32-1:0] in_data_flat;
    logic [N*32-1:0] out_data_flat;

    systolic_sorter #(N) dut (
        .clk(clk),
        .rst(rst),
        .load(load),
        .in_data_flat(in_data_flat),
        .out_data_flat(out_data_flat)
    );

    always #5 clk = ~clk;

    function void apply_input(input logic [31:0] data [N]);
        for (int i = 0; i < N; i++)
            in_data_flat[i*32 +: 32] = data[i];
    endfunction

    function bit check_sorted();
        for (int i = 1; i < N; i++) begin
            if (out_data_flat[(i-1)*32 +: 32] > out_data_flat[i*32 +: 32])
                return 0;
        end
        return 1;
    endfunction

    task run_test(input string label, input logic [31:0] test_data [N]);
        $display("Running test: %s", label);
        apply_input(test_data);
        rst = 1; load = 0; #10;
        rst = 0; load = 1; #10;
        load = 0;
        repeat (N * 2) @(posedge clk);
        if (check_sorted())
            $display("PASS: %s", label);
        else begin
            $display("FAIL: %s", label);
            for (int i = 0; i < N; i++)
                $display("[%0d] = %0d", i, out_data_flat[i*32 +: 32]);
        end
    endtask

    initial begin
        logic [31:0] test1 [N] = '{8, 7, 6, 5, 4, 3, 2, 1};
        logic [31:0] test2 [N] = '{1, 3, 5, 7, 2, 4, 6, 8};
        logic [31:0] test3 [N] = '{9, 2, 6, 3, 7, 5, 1, 4};

        run_test("Descending Order", test1);
        run_test("Alternating High-Low", test2);
        run_test("Random Mix", test3);

        $finish;
    end
endmodule
