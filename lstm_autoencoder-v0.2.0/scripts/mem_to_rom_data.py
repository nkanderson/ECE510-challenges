with open("enc1_U_i.mem") as f:
    hex_words = [line.strip() for line in f if line.strip()]

with open("enc1_U_i.data", "w") as f:
    f.write("".join(hex_words))
