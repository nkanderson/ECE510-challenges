module matvec_tm_multiplier #(
    parameter MAX_ROWS = 64,
    parameter MAX_COLS = 64,
    parameter BANDWIDTH = 16
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    // Matrix dimensions
    input  logic [$clog2(MAX_ROWS)-1:0] num_rows,
    input  logic [$clog2(MAX_COLS)-1:0] num_cols,

    // Vector chunk input
    input  logic vector_write_enable,
    input  logic [$clog2(MAX_COLS)-1:0] vector_base_addr,
    input  logic signed [15:0] vector_in [0:BANDWIDTH-1], // Q4.12

    // Matrix SRAM interface
    output logic [$clog2(MAX_ROWS*MAX_COLS)-1:0] matrix_addr,
    input  logic signed [15:0] matrix_data [0:BANDWIDTH-1], // Q2.14

    // Result output
    output logic signed [15:0] result_out, // Q4.12
    output logic result_valid,
    output logic busy
);

    typedef enum logic [4:0] {
        S_IDLE    = 5'b00001,
        S_VLOAD   = 5'b00010,
        S_LOAD    = 5'b00100,
        S_COMPUTE = 5'b01000,
        S_DONE    = 5'b10000
    } state_t;

    state_t state, next_state;

    logic [$clog2(MAX_ROWS)-1:0] row_idx;
    logic [$clog2(MAX_COLS)-1:0] col_idx;
    logic [$clog2(MAX_COLS)-1:0] chunk_size;
    logic [$clog2(MAX_COLS)-1:0] compute_col;
    logic [$clog2(MAX_COLS)-1:0] vector_load_count;
    
    logic signed [31:0] acc;
    logic signed [15:0] vector_buffer [0:MAX_COLS-1];

    assign busy = (state != S_IDLE);

    // FSM transitions
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:    if (start) next_state = S_VLOAD;
            S_VLOAD:   if (vector_load_count >= num_cols) next_state = S_LOAD;
            S_LOAD:    next_state = S_COMPUTE;
            S_COMPUTE: if (compute_col >= num_cols) next_state = S_DONE;
            S_DONE:    if (row_idx + 1 >= num_rows) next_state = S_IDLE;
                       else next_state = S_LOAD;
        endcase
    end

    // Vector write
    always_ff @(posedge clk) begin
        if (vector_write_enable) begin
            for (int i = 0; i < BANDWIDTH; i++) begin
                if (vector_base_addr + i < MAX_COLS)
                    vector_buffer[vector_base_addr + i] <= vector_in[i];
            end
        end
    end

    // Control counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_idx <= 0;
            col_idx <= 0;
            compute_col <= 0;
            vector_load_count <= 0;
            acc <= 0;
        end else begin
            case (state)
                S_VLOAD: begin
                    if (vector_write_enable)
                        vector_load_count <= vector_load_count + BANDWIDTH;
                end
                S_LOAD: begin
                    col_idx <= 0;
                    acc <= 0;
                end
                S_COMPUTE: begin
                    for (int i = 0; i < BANDWIDTH; i++) begin
                        if (compute_col + i < num_cols) begin
                            acc <= acc + matrix_data[i] * vector_buffer[compute_col + i];
                        end
                    end
                    compute_col <= compute_col + BANDWIDTH;
                end
                S_DONE: begin
                    row_idx <= row_idx + 1;
                    compute_col <= 0;
                end
            endcase
        end
    end

    // Matrix address generation
    assign matrix_addr = row_idx * num_cols + compute_col;

    // Output logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_valid <= 1'b0;
            result_out <= 16'b0;
        end else if (state == S_DONE) begin
            result_valid <= 1'b1;
            result_out <= acc >>> 14; // Truncate Q6.26 to Q4.12
        end else begin
            result_valid <= 1'b0;
        end
    end

endmodule
