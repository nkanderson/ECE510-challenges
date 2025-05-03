module systolic_sorter #(parameter N = 8) (
    input  logic              clk,
    input  logic              rst,
    input  logic              load,
    input  logic [N*32-1:0]   in_data_flat,
    output logic [N*32-1:0]   out_data_flat
);

    logic [31:0] pe_inputs [N];
    logic [31:0] unpacked_input [N];
    logic [31:0] pe_outputs [N];
    logic [31:0] gen_left [N-1];
    logic [31:0] gen_right [N-1];
    logic        phase;

    // Unpack input data into array
    always_comb begin
        for (int i = 0; i < N; i++)
            unpacked_input[i] = in_data_flat[i*32 +: 32];
    end

    // Register for current state
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < N; i++)
                pe_inputs[i] <= 0;
            phase <= 0;
        end else if (load) begin
            for (int i = 0; i < N; i++)
                pe_inputs[i] <= unpacked_input[i];
            phase <= 0;
        end else begin
            for (int i = 0; i < N; i++)
                pe_inputs[i] <= pe_outputs[i];
            phase <= ~phase;
        end
    end

    // Instantiate PEs and wire outputs to temp arrays
    generate
        for (genvar i = 0; i < N-1; i++) begin : gen_pe
            pe pe_inst (
                .clk(clk),
                .rst(rst),
                .in_left(pe_inputs[i]),
                .in_right(pe_inputs[i+1]),
                .sel((phase == 0 && i % 2 == 0) || (phase == 1 && i % 2 == 1)),
                .out_left(gen_left[i]),
                .out_right(gen_right[i])
            );
        end
    endgenerate

    // Determine final outputs based on phase
    always_comb begin
        for (int i = 0; i < N; i++)
            pe_outputs[i] = pe_inputs[i]; // default pass-through

        for (int i = 0; i < N-1; i++) begin
            if ((phase == 0 && i % 2 == 0) || (phase == 1 && i % 2 == 1)) begin
                pe_outputs[i]   = gen_left[i];
                pe_outputs[i+1] = gen_right[i];
            end
        end
    end

    // Output packing
    always_comb begin
        for (int i = 0; i < N; i++)
            out_data_flat[i*32 +: 32] = pe_inputs[i];
    end
endmodule
