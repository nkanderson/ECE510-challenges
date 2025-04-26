// -----------------------------------------------------------------------------
// lstm_cell.sv - LSTM cell
// -----------------------------------------------------------------------------
module lstm_cell #(
    parameter int INPUT_SIZE  = 6,
    parameter int HIDDEN_SIZE = 32,
    parameter int DATA_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,

    // Control
    input  logic input_valid,
    output logic output_valid,

    // Input vectors
    input  logic signed [DATA_WIDTH-1:0] x[INPUT_SIZE],
    input  logic signed [DATA_WIDTH-1:0] h_prev[HIDDEN_SIZE],
    input  logic signed [DATA_WIDTH-1:0] c_prev[HIDDEN_SIZE],

    // Output vectors
    output logic signed [DATA_WIDTH-1:0] h[HIDDEN_SIZE],
    output logic signed [DATA_WIDTH-1:0] c[HIDDEN_SIZE]
);

    // -------------------------------------------------------------------------
    // State Machine
    typedef enum logic [2:0] {
        IDLE,
        LOAD_INPUTS,
        COMPUTE_Wx_Uh,
        ACTIVATE_GATES,
        UPDATE_STATE,
        OUTPUT_READY
    } fsm_state_t;

    fsm_state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // FSM Next State Logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (input_valid) next_state = LOAD_INPUTS;
            LOAD_INPUTS: next_state = COMPUTE_Wx_Uh;
            COMPUTE_Wx_Uh: if (matvec_done) next_state = ACTIVATE_GATES;
            ACTIVATE_GATES: if (activation_done) next_state = UPDATE_STATE;
            UPDATE_STATE: if (update_done) next_state = OUTPUT_READY;
            OUTPUT_READY: if (clear_done) next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Control Signals
    logic load_inputs;
    logic start_matvec, matvec_done;
    logic start_activation, activation_done;
    logic start_update, update_done;
    logic clear_done;

    assign load_inputs     = (state == LOAD_INPUTS);
    assign start_matvec    = (state == COMPUTE_Wx_Uh);
    assign start_activation= (state == ACTIVATE_GATES);
    assign start_update    = (state == UPDATE_STATE);
    assign output_valid    = (state == OUTPUT_READY);

    // -------------------------------------------------------------------------
    // Internal Registers (latching inputs)
    logic signed [DATA_WIDTH-1:0] x_reg[INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] h_prev_reg[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] c_prev_reg[HIDDEN_SIZE];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset registers
        end else if (load_inputs) begin
            x_reg <= x;
            h_prev_reg <= h_prev;
            c_prev_reg <= c_prev;
        end
    end

    // -------------------------------------------------------------------------
    // Placeholder Module Instances (one for each gate)

    // Example: Instantiating one matvec for input gate (i)
    logic signed [DATA_WIDTH-1:0] Wx_i_out[HIDDEN_SIZE];

    matvec_mul #(
        .VEC_SIZE(INPUT_SIZE),
        .OUT_SIZE(HIDDEN_SIZE)
    ) matvec_Wx_i (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_matvec),
        .done(matvec_done),
        .matrix(W_i),
        .vector(x_reg),
        .result(Wx_i_out)
    );

    // TODO: Repeat for Uh_i, Wx_f, Uh_f, Wx_o, Uh_o, Wx_c, Uh_c
    // TODO: Bias addition modules
    // TODO: Activation modules (sigmoid/tanh approximators)

    // -------------------------------------------------------------------------
    // Compute gates i, f, o, and c~ (using activations)

    // Compute new c and new h
    // (Placeholder logic, to be wired fully with elementwise_mul and elementwise_add)

endmodule
// -----------------------------------------------------------------------------
