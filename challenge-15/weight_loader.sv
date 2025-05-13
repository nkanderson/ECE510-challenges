module weight_loader #(parameter BANK_WIDTH = 16, parameter BANK_DEPTH = 65536) (
    output logic [BANK_WIDTH-1:0] sram [0:13][0:BANK_DEPTH-1]
);

    // enc1
    initial begin
        $readmemh("enc1_W_i.mem", sram[0], 0);
        $readmemh("enc1_W_f.mem", sram[0], 384);
        $readmemh("enc1_W_c.mem", sram[0], 768);
        $readmemh("enc1_W_o.mem", sram[0], 1152);

        $readmemh("enc1_U_i.mem", sram[1], 0);
        $readmemh("enc1_U_f.mem", sram[1], 4096);
        $readmemh("enc1_U_c.mem", sram[1], 8192);
        $readmemh("enc1_U_o.mem", sram[1], 12288);

        $readmemh("enc1_b_i.mem", sram[2], 0);
        $readmemh("enc1_b_f.mem", sram[2], 64);
        $readmemh("enc1_b_c.mem", sram[2], 128);
        $readmemh("enc1_b_o.mem", sram[2], 192);
    end

    // enc2
    initial begin
        $readmemh("enc2_W_i.mem", sram[3], 0);
        $readmemh("enc2_W_f.mem", sram[3], 2048);
        $readmemh("enc2_W_c.mem", sram[3], 4096);
        $readmemh("enc2_W_o.mem", sram[3], 6144);

        $readmemh("enc2_U_i.mem", sram[4], 0);
        $readmemh("enc2_U_f.mem", sram[4], 1024);
        $readmemh("enc2_U_c.mem", sram[4], 2048);
        $readmemh("enc2_U_o.mem", sram[4], 3072);

        $readmemh("enc2_b_i.mem", sram[5], 0);
        $readmemh("enc2_b_f.mem", sram[5], 32);
        $readmemh("enc2_b_c.mem", sram[5], 64);
        $readmemh("enc2_b_o.mem", sram[5], 96);
    end

    // dec1
    initial begin
        $readmemh("dec1_W_i.mem", sram[6], 0);
        $readmemh("dec1_W_f.mem", sram[6], 1024);
        $readmemh("dec1_W_c.mem", sram[6], 2048);
        $readmemh("dec1_W_o.mem", sram[6], 3072);

        $readmemh("dec1_U_i.mem", sram[7], 0);
        $readmemh("dec1_U_f.mem", sram[7], 1024);
        $readmemh("dec1_U_c.mem", sram[7], 2048);
        $readmemh("dec1_U_o.mem", sram[7], 3072);

        $readmemh("dec1_b_i.mem", sram[8], 0);
        $readmemh("dec1_b_f.mem", sram[8], 32);
        $readmemh("dec1_b_c.mem", sram[8], 64);
        $readmemh("dec1_b_o.mem", sram[8], 96);
    end

    // dec2
    initial begin
        $readmemh("dec2_W_i.mem", sram[9], 0);
        $readmemh("dec2_W_f.mem", sram[9], 2048);
        $readmemh("dec2_W_c.mem", sram[9], 4096);
        $readmemh("dec2_W_o.mem", sram[9], 6144);

        $readmemh("dec2_U_i.mem", sram[10], 0);
        $readmemh("dec2_U_f.mem", sram[10], 4096);
        $readmemh("dec2_U_c.mem", sram[10], 8192);
        $readmemh("dec2_U_o.mem", sram[10], 12288);

        $readmemh("dec2_b_i.mem", sram[11], 0);
        $readmemh("dec2_b_f.mem", sram[11], 64);
        $readmemh("dec2_b_c.mem", sram[11], 128);
        $readmemh("dec2_b_o.mem", sram[11], 192);
    end

    // Dense layer
    initial begin
        $readmemh("dense_W.mem", sram[12]);
        $readmemh("dense_b.mem", sram[13]);
    end

endmodule
