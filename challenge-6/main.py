from perceptron import Perceptron


def train_nand():
    print("\nTraining for NAND")
    nand_data = [
        ([0, 0], 1),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]
    p = Perceptron(2, learning_rate=0.1)
    p.train(nand_data)
    for x in nand_data:
        print(f"Input: {x[0]}, Predicted: {p.predict(x[0])}, Expected: {x[1]}")


def train_xor():
    print("\nTraining for XOR")
    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]
    p = Perceptron(2, learning_rate=0.1)
    p.train(xor_data)
    for x in xor_data:
        print(f"Input: {x[0]}, Predicted: {p.predict(x[0])}, Expected: {x[1]}")


if __name__ == "__main__":
    train_nand()
    train_xor()
