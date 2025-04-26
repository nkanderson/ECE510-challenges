// -----------------------------------------------------------------------------
// matvec_mul.sv - Single-MAC matrix-vector multiplier
// -----------------------------------------------------------------------------
module matvec_mul #(
    parameter int VEC_SIZE  = 6,
    parameter int OUT_SIZE  = 32,
    parameter int DATA_WIDTH = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    // Matrix: OUT_SIZE rows x VEC_SIZE columns
    input  logic signed [DATA_WIDTH-1:0] matrix[OUT_SIZE][VEC_SIZE],

    // Vector: VEC_SIZE elements
    input  logic signed [DATA_WIDTH-1:0] vector[VEC_SIZE],

    // Result: OUT_SIZE elements
    output logic signed [DATA_WIDTH-1:0] result[OUT_SIZE]
);

    // -------------------------------------------------------------------------
    // Internal State
    typedef enum logic [2:0] {
        IDLE,
        LOAD_ROW,
        MAC_LOOP,
        WRITE_RESULT,
        DONE
    } state_t;

    state_t state, next_state;

    logic signed [DATA_WIDTH-1:0] acc;
    int row_idx;
    int col_idx;

    // -------------------------------------------------------------------------
    // FSM: State Transition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // FSM: Next State Logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD_ROW;
            end

            LOAD_ROW: next_state = MAC_LOOP;

            MAC_LOOP: begin
                if (col_idx == VEC_SIZE-1)
                    next_state = WRITE_RESULT;
            end

            WRITE_RESULT: begin
                if (row_idx == OUT_SIZE-1)
                    next_state = DONE;
                else
                    next_state = LOAD_ROW;
            end

            DONE: begin
                if (!start)
                    next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Counters and Accumulator
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_idx <= 0;
            col_idx <= 0;
            acc <= 0;
        end else begin
            case (state)
                IDLE: begin
                    row_idx <= 0;
                    col_idx <= 0;
                    acc <= 0;
                end

                LOAD_ROW: begin
                    col_idx <= 0;
                    acc <= 0;
                end

                MAC_LOOP: begin
                    acc <= acc + matrix[row_idx][col_idx] * vector[col_idx];
                    if (col_idx < VEC_SIZE-1)
                        col_idx <= col_idx + 1;
                end

                WRITE_RESULT: begin
                    result[row_idx] <= acc;
                    row_idx <= row_idx + 1;
                end

                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Done signal
    assign done = (state == DONE);

endmodule
// -----------------------------------------------------------------------------
