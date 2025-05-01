module systolic_sorter #(parameter N = 8) (
    input  logic              clk,
    input  logic              rst,
    input  logic              load,
    input  logic [N*32-1:0]   in_data_flat,
    output logic [N*32-1:0]   out_data_flat
);
    logic [31:0] values [N];
    logic [31:0] temp_values [N];
    logic phase;

    // Flattened input unpacking
    always_comb begin
        for (int i = 0; i < N; i++)
            temp_values[i] = in_data_flat[i*32 +: 32];
    end

    // Sequential sorting logic
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < N; i++)
                values[i] <= 0;
            phase <= 0;
        end else if (load) begin
            for (int i = 0; i < N; i++)
                values[i] <= temp_values[i];
            phase <= 0;
        end else begin
            logic [31:0] next_values [N];
            for (int i = 0; i < N; i++)
                next_values[i] = values[i];

            for (int i = 0; i < N-1; i++) begin
                if ((phase == 0 && i % 2 == 0) || (phase == 1 && i % 2 == 1)) begin
                    if (values[i] > values[i+1]) begin
                        next_values[i]   = values[i+1];
                        next_values[i+1] = values[i];
                    end
                end
            end
            for (int i = 0; i < N; i++)
                values[i] <= next_values[i];

            phase <= ~phase;
        end
    end

    // Flatten output
    always_comb begin
        for (int i = 0; i < N; i++)
            out_data_flat[i*32 +: 32] = values[i];
    end
endmodule
