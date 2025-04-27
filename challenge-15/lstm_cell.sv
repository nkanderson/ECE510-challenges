module lstm_cell #(
    parameter int INPUT_SIZE  = 2,
    parameter int HIDDEN_SIZE = 2,
    parameter int DATA_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic signed [DATA_WIDTH-1:0] x[INPUT_SIZE],
    input  logic signed [DATA_WIDTH-1:0] h_prev[HIDDEN_SIZE],
    input  logic signed [DATA_WIDTH-1:0] c_prev[HIDDEN_SIZE],

    input  logic signed [DATA_WIDTH-1:0] W[HIDDEN_SIZE*4][INPUT_SIZE],
    input  logic signed [DATA_WIDTH-1:0] U[HIDDEN_SIZE*4][HIDDEN_SIZE],
    input  logic signed [DATA_WIDTH-1:0] b[HIDDEN_SIZE*4],

    output logic signed [DATA_WIDTH-1:0] h[HIDDEN_SIZE],
    output logic signed [DATA_WIDTH-1:0] c[HIDDEN_SIZE]
);

    // FSM States (One-hot encoding)
    typedef enum logic [6:0] {
        S_IDLE      = 7'b0000001,
        S_WX        = 7'b0000010,
        S_UH        = 7'b0000100,
        S_ADD_BIAS  = 7'b0001000,
        S_ACT       = 7'b0010000,
        S_UPDATE    = 7'b0100000,
        S_DONE      = 7'b1000000
    } state_t;

    state_t state, next_state;

    // Matvec control
    logic start_Wx, done_Wx;
    logic start_Uh, done_Uh;

    // Intermediate signals
    logic signed [DATA_WIDTH-1:0] Wx_total[HIDDEN_SIZE*4];
    logic signed [DATA_WIDTH-1:0] Uh_total[HIDDEN_SIZE*4];
    logic signed [DATA_WIDTH-1:0] preact[HIDDEN_SIZE*4];

    logic signed [DATA_WIDTH-1:0] gate_i[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] gate_f[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] gate_c[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] gate_o[HIDDEN_SIZE];
    logic signed [DATA_WIDTH-1:0] tanh_c[HIDDEN_SIZE];

    // FSM sequential
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // FSM combinational
    always_comb begin
        next_state = state;
        start_Wx = 1'b0;
        start_Uh = 1'b0;

        case (state)
            S_IDLE:
                if (start) next_state = S_WX;

            S_WX: begin
                start_Wx = 1'b1;
                if (done_Wx) next_state = S_UH;
            end

            S_UH: begin
                start_Uh = 1'b1;
                if (done_Uh) next_state = S_ADD_BIAS;
            end

            S_ADD_BIAS:
                next_state = S_ACT;  // purely combinational

            S_ACT:
                next_state = S_UPDATE;  // purely combinational

            S_UPDATE:
                next_state = S_DONE;  // purely combinational

            S_DONE:
                next_state = S_IDLE;

            default:
                next_state = S_IDLE;
        endcase
    end

    // Done signal
    assign done = (state == S_DONE);

    // Instantiate Wx and Uh matvec multipliers
    matvec_mul #(
        .VEC_SIZE(INPUT_SIZE),
        .OUT_SIZE(HIDDEN_SIZE*4),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_matvec_Wx (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_Wx),
        .done(done_Wx),
        .matrix(W),
        .vector(x),
        .result(Wx_total)
    );

    matvec_mul #(
        .VEC_SIZE(HIDDEN_SIZE),
        .OUT_SIZE(HIDDEN_SIZE*4),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_matvec_Uh (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_Uh),
        .done(done_Uh),
        .matrix(U),
        .vector(h_prev),
        .result(Uh_total)
    );

    // Combinational elementwise add: preact = Wx_total + Uh_total + b
    always_comb begin
        integer i;
        for (i = 0; i < HIDDEN_SIZE*4; i++) begin
            preact[i] = Wx_total[i] + Uh_total[i] + b[i];
        end
    end

    // Instantiate activation functions
    generate
        genvar idx;
        for (idx = 0; idx < HIDDEN_SIZE; idx++) begin
            sigmoid_approx u_sigmoid_i (.x(preact[idx]),                  .y(gate_i[idx]));
            sigmoid_approx u_sigmoid_f (.x(preact[HIDDEN_SIZE+idx]),       .y(gate_f[idx]));
            tanh_approx    u_tanh_c     (.x(preact[HIDDEN_SIZE*2+idx]),    .y(gate_c[idx]));
            sigmoid_approx u_sigmoid_o (.x(preact[HIDDEN_SIZE*3+idx]),     .y(gate_o[idx]));
            tanh_approx    u_tanh_c2    (.x(c[idx]),                       .y(tanh_c[idx])); // tanh(c) for output
        end
    endgenerate

    // Update h and c
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            foreach (h[i]) h[i] <= 0;
            foreach (c[i]) c[i] <= 0;
        end else if (state == S_UPDATE) begin
            integer i;
            for (i = 0; i < HIDDEN_SIZE; i++) begin
                c[i] <= (gate_f[i] * c_prev[i]) + (gate_i[i] * gate_c[i]);
                h[i] <= gate_o[i] * tanh_c[i];
            end
        end
    end

endmodule
