module QUpdateUnit (
    input  logic clk,
    input  logic rst,
    input  logic [15:0] Q_old,        // Q(s,a)
    input  logic [15:0] Q_next_max,   // max_a' Q(s',a')
    input  logic [7:0]  reward,       // Integer reward
    input  logic start,
    output logic [15:0] Q_updated,
    output logic done
);

    // Fixed-point constants (Q8.8 format)
    parameter logic [15:0] ALPHA     = 16'h0080;  // 0.5
    parameter logic [15:0] ONE_MINUS_ALPHA = 16'h0080;  // 1 - alpha
    parameter logic [15:0] GAMMA     = 16'h0E66;  // ~0.9 in Q8.8

    // Internal registers
    logic [31:0] temp1, temp2, temp3;
    logic [15:0] reward_q88;
    logic busy;

    assign done = ~busy;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            Q_updated <= 0;
            busy <= 0;
        end else if (start) begin
            busy <= 1;

            // Convert reward (8-bit int) to Q8.8
            reward_q88 = {reward, 8'b0};  

            // Compute reward + gamma * Q_next_max
            temp1 = reward_q88 + ((Q_next_max * GAMMA) >> 8); // Q8.8

            // Multiply by alpha
            temp2 = (ALPHA * temp1) >> 8; // Q8.8

            // Compute (1 - alpha) * Q_old
            temp3 = (ONE_MINUS_ALPHA * Q_old) >> 8;

            // Sum to get updated Q
            Q_updated <= temp2[15:0] + temp3[15:0];
            busy <= 0;
        end
    end
endmodule
