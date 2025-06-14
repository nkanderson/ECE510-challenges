//------------------------------------------------------------------------------
// File      : matvec_mult.sv
// Author    : Niklas Anderson with ChatGPT assistance
// Project   : ECE510 Challenge 10
// Created   : Spring 2025
// Description : Matrix-vector multiplication module
//              This module performs matrix-vector multiplication using a single MAC unit.
//              It supports a fixed-point representation for both matrix and vector inputs.
//------------------------------------------------------------------------------

module matvec_mult #(
    parameter MAX_ROWS = 64,
    parameter MAX_COLS = 64,
    parameter BANDWIDTH = 16,
    parameter DATA_WIDTH = 16,
    parameter VEC_NUM_BANKS  = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    // Matrix dimensions
    input  logic [$clog2(MAX_ROWS):0] num_rows,
    input  logic [$clog2(MAX_COLS):0] num_cols,

    // Vector chunk input
    // NOTE: Client is required to convert vector values to Q4.12
    input  logic vector_write_enable,
    input  logic [$clog2(MAX_COLS)-1:0] vector_base_addr,
    input  wire logic signed [DATA_WIDTH*BANDWIDTH-1:0] vector_in, // Q4.12

    // Matrix SRAM interface
    output logic [$clog2(MAX_ROWS*MAX_COLS)-1:0] matrix_addr,
    output logic matrix_enable,
    input  wire logic [DATA_WIDTH*BANDWIDTH-1:0] matrix_data, // Q2.14
    input  logic matrix_ready,

    // Result output
    output logic signed [(DATA_WIDTH*2)-1:0] result_out, // Q20.12 (see note below)
    output logic result_valid,
    output logic busy
);

    localparam VEC_BANK_WIDTH = MAX_COLS / VEC_NUM_BANKS;

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

    logic signed [(DATA_WIDTH*2):0] acc;
    logic [VEC_BANK_WIDTH*DATA_WIDTH-1:0] vector_bank [VEC_NUM_BANKS-1:0];

    // Vector load control
    logic vector_loaded;

    // MAC calculation and result
    logic signed [(DATA_WIDTH*2)-1:0] mac_result;
    logic [DATA_WIDTH*MAC_CHUNK_SIZE-1:0] a;
    logic [DATA_WIDTH*MAC_CHUNK_SIZE-1:0] b;

    // TODO: Consider using multiple MACs in parallel to improve throughput
    mac4 mac(
              .clk(clk),
              .a0(a[0*DATA_WIDTH +: DATA_WIDTH]), .b0(b[0*DATA_WIDTH +: DATA_WIDTH]),
              .a1(a[1*DATA_WIDTH +: DATA_WIDTH]), .b1(b[1*DATA_WIDTH +: DATA_WIDTH]),
              .a2(a[2*DATA_WIDTH +: DATA_WIDTH]), .b2(b[2*DATA_WIDTH +: DATA_WIDTH]),
              .a3(a[3*DATA_WIDTH +: DATA_WIDTH]), .b3(b[3*DATA_WIDTH +: DATA_WIDTH]),
              .result(mac_result)
            );

    //
    // FSM next state update and state transitions
    //
    always_ff @(posedge clk) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // FSM next state logic
    always @(*) begin
        next_state = state;
        unique case (state)
            S_IDLE:      if (start) next_state = S_VLOAD;
            S_VLOAD:     if (vector_loaded) next_state = S_REQ_MAT;
            S_REQ_MAT:   next_state = S_WAIT_MAT;
            // TODO: We should ideally start mutiplying while we are streaming in
            // matrix data, since we can't get more than BANDWIDTH values at once.
            S_WAIT_MAT:  if (matrix_ready) next_state = S_MAC;
            // From S_MAC: If done with the current matrix chunk, either we need another chunk
            // or we're done (if we're at the last row).
            S_MAC:       if (num_ops + MAC_CHUNK_SIZE >= BANDWIDTH)
                             next_state = (row_idx < num_rows) ? S_REQ_MAT : S_DONE;
                         else
                             next_state = S_MAC;
            S_DONE:      next_state = S_IDLE;
            default:     next_state = state;
        endcase
    end
    // End FSM logic

    //
    // Vector loading logic
    //
    // Get the entire vector from the client before moving forward with
    // chunked reading of matrix and MAC operation.
    // The client needs to hold vector_write_enable high and increment
    // the vector_base_addr while updating vector_in. While vector_write_enable
    // is high, this module will continue reading in vector chunks.
    always_ff @(posedge clk) begin
        integer i;
        if (!rst_n) begin
            for (i = 0; i < MAX_COLS; i++) begin
                vector_bank[i] <= 0;
            end
        end else begin
            if (vector_write_enable) begin
              for (i = 0; i < BANDWIDTH; i++) begin
                  // TODO: Consider pre-checking bank bounds, e.g. if (vector_base_addr + BANDWIDTH < num_cols)
                  // Get top bits to index into the vector bank
                  // and the lower bits to index into the DATA_WIDTH segment using modulo logic
                  vector_bank[
                    (vector_base_addr + i) >> $clog2(VEC_BANK_WIDTH)
                  ][
                    ((vector_base_addr + i) & (VEC_BANK_WIDTH - 1))*DATA_WIDTH +: DATA_WIDTH
                  ] <= vector_in[i*DATA_WIDTH +: DATA_WIDTH];
              end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            vector_loaded <= 0;
        end else begin
            if (vector_write_enable) begin
                // Check if we have loaded the entire vector
                if (vector_base_addr + BANDWIDTH >= num_cols)
                    vector_loaded <= 1;
                else
                    vector_loaded <= 0;
            end else if (state == S_IDLE) begin
              vector_loaded <= 0;
            end else begin
              // Keep vector_loaded unchanged in other states
              // vector_loaded <= vector_loaded;
              // Hold value (should be done implicitly without the above explicit assignment)
            end
        end
    end
    // End vector loading logic

    // Matrix fetch signals
    // TODO: We should clock in matrix_data so we're not relying on matrix_loader
    // to hold the data on the wires. This would allow for beginning multiplication
    // while continuing to retrieve more matrix data.
    assign matrix_enable = (state == S_REQ_MAT || state == S_WAIT_MAT);
    assign matrix_addr   = row_idx * num_cols + col_idx;

    //
    // MAC accumulation with row tracking
    //
    // row and col index updates
    reg [$clog2(MAX_ROWS)-1:0] row_idx_next;
    reg [$clog2(MAX_COLS)-1:0] col_idx_next;
    always @(*) begin
      row_idx_next = row_idx;
      col_idx_next = col_idx;

      if (state == S_MAC) begin
          // If the next MAC operation will be the last for this row, update the
          // row_idx and col_idx on the next clock cycle. Do this until the final
          // increment of row_idx will reach the last row.
          if (col_idx + MAC_CHUNK_SIZE >= num_cols && row_idx < (num_rows - 1)) begin
              row_idx_next = row_idx + 1;
              col_idx_next = 0;
          end else begin
              col_idx_next = col_idx + MAC_CHUNK_SIZE;
          end
      end else if (state == S_IDLE) begin
          col_idx_next = 0;
          row_idx_next = 0;
      end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            row_idx <= 0;
            col_idx <= 0;
        end else begin
            row_idx <= row_idx_next;
            col_idx <= col_idx_next;
        end
    end

    //
    // num_ops updates
    //
    logic [BANDWIDTH-1:0] num_ops_next;
    always @(*) begin
        if (state == S_MAC) begin
            // Increment num_ops by MAC_CHUNK_SIZE for each MAC operation
            num_ops_next = num_ops + MAC_CHUNK_SIZE;
        end else if (state == S_WAIT_MAT || state == S_IDLE) begin
            // Reset num_ops when waiting for matrix data or when idle
            num_ops_next = 0;
        end else begin
            // Keep num_ops unchanged in other states
            num_ops_next = num_ops;
        end
    end

    always @(posedge clk) begin
        if (!rst_n)
            num_ops <= 0;
        else
            num_ops <= num_ops_next;
    end

    //
    // acc updates
    //
    logic signed [(DATA_WIDTH*2):0] acc_next;
    always @(*) begin
        if (col_idx + MAC_CHUNK_SIZE >= num_cols) begin
            // If the current MAC is the last, reset the acc next cycle
            acc_next = 0;
        end else if (state == S_MAC) begin
            acc_next = acc + mac_result;
        end else begin
            // Keep acc unchanged in other states
            acc_next = acc;
        end
    end

    always @(posedge clk) begin
        if (!rst_n)
            acc <= 0;
        else
            acc <= acc_next;
    end
    // End MAC accumulation, row and column tracking, and num_ops updates

    // MAC result
    generate
        for (genvar i = 0; i < MAC_CHUNK_SIZE; i++) begin : mac_inputs
            // Logically, we can think about using the following values:
            // Global column index
            // int total_col_idx = col_idx + i;
            // Bank and local index:
            // int bank_idx  = total_col_idx >> $clog2(VEC_BANK_WIDTH);
            // int local_idx = total_col_idx & (VEC_BANK_WIDTH - 1);
            // But since col_idx in a runtime variable, we can't use localparam with the above computations.
            // So instead, we're inlining the expressions, but it may be helpful to reference the above values
            // for understanding the logic.

            // Since we have the full vector, we can simply use the col_idx here. matrix_data is used in
            // BANDWIDTH-sized chunks, so we need to modulo the col_idx by BANDWIDTH to get the index for
            // the current chunk.
            assign a[i*DATA_WIDTH +: DATA_WIDTH] =
                (state == S_MAC && ((col_idx + i) < num_cols))
                    ? matrix_data[((col_idx % BANDWIDTH) + i) * DATA_WIDTH +: DATA_WIDTH]
                    : '0;
            assign b[i*DATA_WIDTH +: DATA_WIDTH] =
                (state == S_MAC && (col_idx + i) < num_cols)
                    ? vector_bank[
                        (col_idx + i) >> $clog2(VEC_BANK_WIDTH)
                      ][
                        ((col_idx + i) & (VEC_BANK_WIDTH - 1)) * DATA_WIDTH +: DATA_WIDTH
                      ]
                    : '0;
        end
    endgenerate

    //
    // Output logic
    //
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            result_out   <= 0;
            result_valid <= 0;
        // Stream out the result after each row is completed
        end else if (state == S_MAC &&
                    col_idx + MAC_CHUNK_SIZE >= num_cols) begin
            // TODO: Decide if this format should be adjusted - right now it allows
            // for overflow of the Q4.12 fixed point. This final format is Q20.12.
            // We may want to saturate or truncate instead, but in that case
            // we need to handle overflow and signedness appropriately.
            result_out   <= acc + mac_result;
            result_valid <= 1;
        end else begin
            result_valid <= 0;
        end
    end

    assign busy = (state != S_IDLE);
    // End output logic

    // function automatic logic signed [DATA_WIDTH-1:0] get_vector_value(input int col_idx);
    //     int bank_id = col_idx / VEC_BANK_WIDTH;
    //     int local_idx = col_idx % VEC_BANK_WIDTH;
    //     return vector_bank[bank_id][local_idx*DATA_WIDTH +: DATA_WIDTH];
    // endfunction


endmodule
