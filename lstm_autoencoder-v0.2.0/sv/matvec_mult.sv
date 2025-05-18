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

    // NOTE: vector_in and matrix_data are defaulting to var rather than wire
    // due to default compile options. Should confirm this is as desired.
    // Vector chunk input
    // NOTE: Client is required to convert vector values to Q4.12
    input  logic vector_write_enable,
    input  logic [$clog2(MAX_COLS)-1:0] vector_base_addr,
    input  logic signed [15:0] vector_in [0:BANDWIDTH-1], // Q4.12

    // Matrix SRAM interface
    output logic [$clog2(MAX_ROWS*MAX_COLS)-1:0] matrix_addr,
    output logic matrix_enable,
    input  logic signed [DATA_WIDTH-1:0] matrix_data [0:BANDWIDTH-1], // Q2.14
    input  logic matrix_ready,

    // Result output
    output logic signed [(DATA_WIDTH*2)-1:0] result_out, // Q20.12 (see note below)
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
    localparam MAC_CHUNK_SIZE = 4;
    logic [BANDWIDTH-1:0] num_ops;

    logic signed [31:0] acc;
    logic signed [DATA_WIDTH-1:0] vector_buffer [0:MAX_COLS-1];

    // Vector load control
    logic vector_loaded;

    // MAC calculation and result
    logic signed [(DATA_WIDTH*2)-1:0] mac_result;
    logic [$clog2(MAC_CHUNK_SIZE)-1:0] i;
    logic [DATA_WIDTH-1:0] a [MAC_CHUNK_SIZE-1:0];
    logic [DATA_WIDTH-1:0] b [MAC_CHUNK_SIZE-1:0];
    logic [$clog2(MAX_ROWS * MAX_COLS)-1:0] mat_idx;
    logic [$clog2(MAX_COLS)-1:0] vec_idx;

    mac4 mac(
              .a0(a[0]), .b0(b[0]),
              .a1(a[1]), .b1(b[1]),
              .a2(a[2]), .b2(b[2]),
              .a3(a[3]), .b3(b[3]),
              .result(mac_result)
            );

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
            S_MAC:       if (col_idx >= num_cols)
                             next_state = (row_idx + 1 < num_rows) ? S_REQ_MAT : S_DONE;
                         else
                             next_state = S_REQ_MAT;
            S_DONE:      next_state = S_IDLE;
        endcase
    end

    // Get the entire vector from the client before moving forward with
    // chunked reading of matrix and MAC operation.
    // The client needs to hold vector_write_enable high and increment
    // the vector_base_addr while updating vector_in. While vector_write_enable
    // is high, this module will continue reading in vector chunks.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vector_loaded <= 0;
        end else if (vector_write_enable) begin
            // TODO: BANDWIDTH is set at instantiation, so we need to make sure having
            // a value of 16 here will work for all of our cases. Maybe we can do some
            // special casing where if num_rows and num_cols are less than BANDWIDTH,
            // we can jump straight to S_MAC state
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
    // TODO: Do we want to clock in matrix_data so we're not relying on matrix_loader
    // to hold the data on the wires?
    assign matrix_enable = (state == S_REQ_MAT || state == S_WAIT_MAT);
    assign matrix_addr   = row_idx * num_cols + col_idx;

    // MAC accumulation with row tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 0;
            col_idx <= 0;
            row_idx <= 0;
            num_ops <= 0;
        end else if (state == S_MAC) begin
            // If the next MAC operation will be the last for this row, update the
            // row_idx and col_idx on the next clock cycle. Do this until the final
            // increment of row_idx will reach the last row.
            if (col_idx + MAC_CHUNK_SIZE >= num_cols && row_idx < (num_rows - 1)) begin
                row_idx <= row_idx + 1;
                col_idx <= 0;
            end else begin
              col_idx <= col_idx + MAC_CHUNK_SIZE;
            end
            acc <= acc + mac_result;
            num_ops <= num_ops + MAC_CHUNK_SIZE;
        end else if (state == S_WAIT_MAT) begin
            num_ops <= 0;
        end else if (state == S_IDLE) begin
            col_idx <= 0;
            row_idx <= 0;
            num_ops <= 0;
        end
    end

    // MAC result
    always_comb begin
      if (state == S_MAC) begin
        for (i = 0; i < MAC_CHUNK_SIZE; i++) begin
            mat_idx = row_idx * num_cols + col_idx + i;
            vec_idx = col_idx + i;

            if ((col_idx + i) < num_cols) begin
                a[i] = matrix_data[mat_idx];
                b[i] = vector_buffer[vec_idx];
            end else begin
                a[i] = 0;
                b[i] = 0;
            end
        end
      end
    end

    // Output logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_out   <= 0;
            result_valid <= 0;
        // When acc has the final valid value clocked in, row_idx should have been held
        // at num_(row - 1). col_idx should increase by MAC_CHUNK_SIZE until the below condition.
        // num_ops increments by MAC_CHUNK_SIZE, so it may overshoot BANDWIDTH if it's not
        // evenly divisible by MAC_CHUNK_SIZE.
        end else if (state == S_MAC &&
                     row_idx == num_rows - 1 &&
                     num_ops >= BANDWIDTH) begin
            // TODO: Decide if this format should be adjusted - right now it allows
            // for overflow of the Q4.12 fixed point. I think this final format then
            // is Q20.12. We may want to saturdate or truncate instead, but in that case
            // we need to handle overflow and signedness appropriately.
            result_out   <= acc;
            result_valid <= 1;
        end else begin
            result_valid <= 0;
        end
    end

endmodule
