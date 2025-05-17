module matvec_multiplier #(
    parameter MAX_ROWS = 64,
    parameter MAX_COLS = 64,
    parameter BANDWIDTH = 16,
    parameter DATA_WIDTH = 16
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
    output logic matrix_enable,
    input  logic signed [DATA_WIDTH-1:0] matrix_data [0:BANDWIDTH-1], // Q2.14
    input  logic matrix_ready,

    // Result output
    output logic signed [DATA_WIDTH-1:0] result_out, // Q4.12
    output logic result_valid,
    output logic busy
);

    typedef enum logic [5:0] {
        S_IDLE      = 6'b000001,
        S_VLOAD     = 6'b000010,
        S_REQ_MAT   = 6'b000100,
        S_WAIT_MAT  = 6'b001000,
        S_MAC       = 6'b010000,
        S_DONE      = 6'b100000
    } state_t;

    state_t state, next_state;

    // Internal loop counters
    logic [$clog2(MAX_ROWS)-1:0] row_idx;
    logic [$clog2(MAX_COLS)-1:0] col_idx;
    
    logic signed [31:0] acc;
    logic signed [DATA_WIDTH-1:0] vector_buffer [0:MAX_COLS-1];

    // Output logic
    assign busy = (state != S_IDLE);

    // FSM transitions
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // FSM next state logic
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:      if (start) next_state = S_VLOAD;
            S_VLOAD:     if (vector_loaded) next_state = S_REQ_MAT;
            S_REQ_MAT:   next_state = S_WAIT_MAT;
            S_WAIT_MAT:  if (matrix_ready) next_state = S_MAC;
            S_MAC:       if (col_idx + BANDWIDTH >= num_cols)
                             next_state = (row_idx + 1 < num_rows) ? S_REQ_MAT : S_DONE;
                         else
                             next_state = S_REQ_MAT;
            S_DONE:      next_state = S_IDLE;
        endcase
    end

    // Vector load control
    logic vector_loaded;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vector_loaded <= 0;
        end else if (vector_write_enable) begin
            for (int i = 0; i < BANDWIDTH; i++) begin
                vector_buffer[vector_base_addr + i] <= vector_in[i];
            end
            if (vector_base_addr + BANDWIDTH >= num_cols)
                vector_loaded <= 1;
        end else if (state == S_IDLE) begin
            vector_loaded <= 0;
        end
    end

    // Matrix fetch signals
    assign matrix_enable = (state == S_REQ_MAT);
    assign matrix_addr   = row_idx * num_cols + col_idx;

    // MAC accumulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 0;
            col_idx <= 0;
        end else if (state == S_MAC) begin
            for (int i = 0; i < BANDWIDTH; i++) begin
                if (col_idx + i < num_cols) begin
                    logic signed [31:0] product = matrix_data[i] * vector_buffer[col_idx + i]; // Q2.14 × Q4.12 = Q6.26
                    acc <= acc + (product >>> 14); // Shift back to Q4.12
                end
            end
            col_idx <= col_idx + BANDWIDTH;
        end else if (state == S_REQ_MAT) begin
            if (col_idx == 0) acc <= 0;
        end else if (state == S_IDLE) begin
            col_idx <= 0;
        end
    end

    // Row tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_idx <= 0;
        end else if (state == S_MAC && col_idx + BANDWIDTH >= num_cols) begin
            row_idx <= row_idx + 1;
            col_idx <= 0;
        end else if (state == S_IDLE) begin
            row_idx <= 0;
        end
    end

    // Output logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_out   <= 0;
            result_valid <= 0;
        end else if (state == S_MAC && col_idx + BANDWIDTH >= num_cols) begin
            result_out   <= acc[27:12]; // Take middle 16 bits for Q4.12
            result_valid <= 1;
        end else begin
            result_valid <= 0;
        end
    end

endmodule
