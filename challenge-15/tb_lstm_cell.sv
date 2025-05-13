`timescale 1ns / 1ps

module tb_lstm_cell;

    parameter int INPUT_SIZE   = 6;
    parameter int HIDDEN1_SIZE = 64;
    parameter int HIDDEN2_SIZE = 32;
    parameter int DATA_WIDTH   = 16;

    logic clk;
    logic rst_n;
    logic start1, done1;
    logic start2, done2;

    // Inputs for LSTM1
    logic signed [DATA_WIDTH-1:0] x_in[INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] h1_prev[HIDDEN1_SIZE];
    logic signed [DATA_WIDTH-1:0] c1_prev[HIDDEN1_SIZE];
    logic signed [DATA_WIDTH-1:0] h1[HIDDEN1_SIZE];
    logic signed [DATA_WIDTH-1:0] c1[HIDDEN1_SIZE];

    // Inputs for LSTM2
    logic signed [DATA_WIDTH-1:0] h2_prev[HIDDEN2_SIZE];
    logic signed [DATA_WIDTH-1:0] c2_prev[HIDDEN2_SIZE];
    logic signed [DATA_WIDTH-1:0] h2[HIDDEN2_SIZE];
    logic signed [DATA_WIDTH-1:0] c2[HIDDEN2_SIZE];

    // LSTM1 weights
    logic signed [DATA_WIDTH-1:0] W1[HIDDEN1_SIZE*4][INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] U1[HIDDEN1_SIZE*4][HIDDEN1_SIZE];
    logic signed [DATA_WIDTH-1:0] b1[HIDDEN1_SIZE*4];

    // LSTM2 weights
    logic signed [DATA_WIDTH-1:0] W2[HIDDEN2_SIZE*4][HIDDEN1_SIZE];
    logic signed [DATA_WIDTH-1:0] U2[HIDDEN2_SIZE*4][HIDDEN2_SIZE];
    logic signed [DATA_WIDTH-1:0] b2[HIDDEN2_SIZE*4];

    // Golden outputs
    logic signed [DATA_WIDTH-1:0] golden_h2[1000][HIDDEN2_SIZE]; // max 1000 steps
    logic signed [DATA_WIDTH-1:0] golden_c2[1000][HIDDEN2_SIZE];

    // Stimuli
    logic signed [DATA_WIDTH-1:0] x_inputs[1000][INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] h2_prev_all[1000][HIDDEN2_SIZE];
    logic signed [DATA_WIDTH-1:0] c2_prev_all[1000][HIDDEN2_SIZE];

    int timestep_count;

    // Clock generation
    always #5 clk = ~clk;

    // Instantiate LSTM1
    lstm_cell #(
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN1_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) lstm1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start1),
        .done(done1),
        .x(x_in),
        .h_prev(h1_prev),
        .c_prev(c1_prev),
        .W(W1),
        .U(U1),
        .b(b1),
        .h(h1),
        .c(c1)
    );

    // Instantiate LSTM2
    lstm_cell #(
        .INPUT_SIZE(HIDDEN1_SIZE),
        .HIDDEN_SIZE(HIDDEN2_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) lstm2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start2),
        .done(done2),
        .x(h1),
        .h_prev(h2_prev),
        .c_prev(c2_prev),
        .W(W2),
        .U(U2),
        .b(b2),
        .h(h2),
        .c(c2)
    );

    initial begin
        clk = 0;
        rst_n = 0;
        start1 = 0;
        start2 = 0;

        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // --- Load input vectors and golden outputs ---
        $readmemh("x_in.mem",     x_inputs);
        $readmemh("h_prev.mem",   h2_prev_all);
        $readmemh("c_prev.mem",   c2_prev_all);
        $readmemh("h_out.mem",    golden_h2);
        $readmemh("c_out.mem",    golden_c2);

        // --- Count timesteps based on file size ---
        timestep_count = 0;
        while (x_inputs[timestep_count][0] !== 'bx) timestep_count++;

        // --- Load LSTM1 Weights ---
        $readmemh("enc1_W_i.mem", W1[0]);
        $readmemh("enc1_W_f.mem", W1[1]);
        $readmemh("enc1_W_c.mem", W1[2]);
        $readmemh("enc1_W_o.mem", W1[3]);

        $readmemh("enc1_U_i.mem", U1[0]);
        $readmemh("enc1_U_f.mem", U1[1]);
        $readmemh("enc1_U_c.mem", U1[2]);
        $readmemh("enc1_U_o.mem", U1[3]);

        $readmemh("enc1_b_i.mem", b1[0]);
        $readmemh("enc1_b_f.mem", b1[1]);
        $readmemh("enc1_b_c.mem", b1[2]);
        $readmemh("enc1_b_o.mem", b1[3]);

        // --- Load LSTM2 Weights ---
        $readmemh("enc2_W_i.mem", W2[0]);
        $readmemh("enc2_W_f.mem", W2[1]);
        $readmemh("enc2_W_c.mem", W2[2]);
        $readmemh("enc2_W_o.mem", W2[3]);

        $readmemh("enc2_U_i.mem", U2[0]);
        $readmemh("enc2_U_f.mem", U2[1]);
        $readmemh("enc2_U_c.mem", U2[2]);
        $readmemh("enc2_U_o.mem", U2[3]);

        $readmemh("enc2_b_i.mem", b2[0]);
        $readmemh("enc2_b_f.mem", b2[1]);
        $readmemh("enc2_b_c.mem", b2[2]);
        $readmemh("enc2_b_o.mem", b2[3]);

        // --- Run through all timesteps ---
        for (int t = 0; t < timestep_count; t++) begin
            $display("[t=%0d] Running LSTM inference step", t);

            // Set inputs
            for (int i = 0; i < INPUT_SIZE; i++) x_in[i] = x_inputs[t][i];
            for (int i = 0; i < HIDDEN1_SIZE; i++) h1_prev[i] = 0;  // initially zero
            for (int i = 0; i < HIDDEN1_SIZE; i++) c1_prev[i] = 0;

            for (int i = 0; i < HIDDEN2_SIZE; i++) h2_prev[i] = h2_prev_all[t][i];
            for (int i = 0; i < HIDDEN2_SIZE; i++) c2_prev[i] = c2_prev_all[t][i];

            // Run LSTM1
            @(posedge clk); start1 = 1;
            @(posedge clk); start1 = 0;
            wait (done1);

            // Run LSTM2
            @(posedge clk); start2 = 1;
            @(posedge clk); start2 = 0;
            wait (done2);

            // Check output
            for (int i = 0; i < HIDDEN2_SIZE; i++) begin
                if (h2[i] !== golden_h2[t][i])
                    $fatal(1, "FAIL: h2[%0d] = %0d, expected %0d", i, h2[i], golden_h2[t][i]);
                if (c2[i] !== golden_c2[t][i])
                    $fatal(1, "FAIL: c2[%0d] = %0d, expected %0d", i, c2[i], golden_c2[t][i]);
            end
        end

        $display("✅ All %0d timesteps passed!", timestep_count);
        $finish;
    end

endmodule
