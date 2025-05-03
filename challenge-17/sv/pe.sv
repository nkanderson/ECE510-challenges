module pe (
    input  logic clk,
    input  logic rst,
    input  logic [31:0] in_left,
    input  logic [31:0] in_right,
    input  logic sel,  // 1 for active this phase
    output logic [31:0] out_left,
    output logic [31:0] out_right
);
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            out_left  <= 0;
            out_right <= 0;
        end else if (sel) begin
            if (in_left > in_right) begin
                out_left  <= in_right;
                out_right <= in_left;
            end else begin
                out_left  <= in_left;
                out_right <= in_right;
            end
        end else begin
            out_left  <= in_left;
            out_right <= in_right;
        end
    end
endmodule
