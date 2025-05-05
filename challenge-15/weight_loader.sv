module weight_loader #(parameter MEM_WIDTH = 16, parameter MEM_DEPTH = 99394) (
    output logic [MEM_WIDTH-1:0] sram [0:MEM_DEPTH-1]
);

    initial $readmemh("enc1_W_i.mem", sram, 0, 383);
    initial $readmemh("enc1_U_i.mem", sram, 384, 4479);
    initial $readmemh("enc1_b_i.mem", sram, 4480, 4543);
    initial $readmemh("enc1_W_f.mem", sram, 4544, 4927);
    initial $readmemh("enc1_U_f.mem", sram, 4928, 9023);
    initial $readmemh("enc1_b_f.mem", sram, 9024, 9087);
    initial $readmemh("enc1_W_c.mem", sram, 9088, 9471);
    initial $readmemh("enc1_U_c.mem", sram, 9472, 13567);
    initial $readmemh("enc1_b_c.mem", sram, 13568, 13631);
    initial $readmemh("enc1_W_o.mem", sram, 13632, 14015);
    initial $readmemh("enc1_U_o.mem", sram, 14016, 18111);
    initial $readmemh("enc1_b_o.mem", sram, 18112, 18175);
    initial $readmemh("enc2_W_i.mem", sram, 18176, 20223);
    initial $readmemh("enc2_U_i.mem", sram, 20224, 21247);
    initial $readmemh("enc2_b_i.mem", sram, 21248, 21279);
    initial $readmemh("enc2_W_f.mem", sram, 21280, 23327);
    initial $readmemh("enc2_U_f.mem", sram, 23328, 24351);
    initial $readmemh("enc2_b_f.mem", sram, 24352, 24383);
    initial $readmemh("enc2_W_c.mem", sram, 24384, 26431);
    initial $readmemh("enc2_U_c.mem", sram, 26432, 27455);
    initial $readmemh("enc2_b_c.mem", sram, 27456, 27487);
    initial $readmemh("enc2_W_o.mem", sram, 27488, 29535);
    initial $readmemh("enc2_U_o.mem", sram, 29536, 30559);
    initial $readmemh("enc2_b_o.mem", sram, 30560, 30591);
    initial $readmemh("dec1_W_i.mem", sram, 30592, 31615);
    initial $readmemh("dec1_U_i.mem", sram, 31616, 32639);
    initial $readmemh("dec1_b_i.mem", sram, 32640, 32671);
    initial $readmemh("dec1_W_f.mem", sram, 32672, 33695);
    initial $readmemh("dec1_U_f.mem", sram, 33696, 34719);
    initial $readmemh("dec1_b_f.mem", sram, 34720, 34751);
    initial $readmemh("dec1_W_c.mem", sram, 34752, 35775);
    initial $readmemh("dec1_U_c.mem", sram, 35776, 36799);
    initial $readmemh("dec1_b_c.mem", sram, 36800, 36831);
    initial $readmemh("dec1_W_o.mem", sram, 36832, 37855);
    initial $readmemh("dec1_U_o.mem", sram, 37856, 38879);
    initial $readmemh("dec1_b_o.mem", sram, 38880, 38911);
    initial $readmemh("dec2_W_i.mem", sram, 38912, 40959);
    initial $readmemh("dec2_U_i.mem", sram, 40960, 45055);
    initial $readmemh("dec2_b_i.mem", sram, 45056, 45119);
    initial $readmemh("dec2_W_f.mem", sram, 45120, 47167);
    initial $readmemh("dec2_U_f.mem", sram, 47168, 51263);
    initial $readmemh("dec2_b_f.mem", sram, 51264, 51327);
    initial $readmemh("dec2_W_c.mem", sram, 51328, 53375);
    initial $readmemh("dec2_U_c.mem", sram, 53376, 57471);
    initial $readmemh("dec2_b_c.mem", sram, 57472, 57535);
    initial $readmemh("dec2_W_o.mem", sram, 57536, 59583);
    initial $readmemh("dec2_U_o.mem", sram, 59584, 63679);
    initial $readmemh("dec2_b_o.mem", sram, 63680, 63743);
    initial $readmemh("dense_W.mem", sram, 63744, 64127);
    initial $readmemh("dense_b.mem", sram, 64128, 64133);

endmodule
