import torch
import unittest


def systolic_bubble_sort(input_tensor, device="mps"):
    x = input_tensor.clone().to(device)
    n = x.shape[0]

    for t in range(n):
        if t % 2 == 0:
            idx1 = torch.arange(0, n - 1, 2, device=device)
        else:
            idx1 = torch.arange(1, n - 1, 2, device=device)
        idx2 = idx1 + 1

        vals1 = x[idx1]
        vals2 = x[idx2]

        min_vals = torch.minimum(vals1, vals2)
        max_vals = torch.maximum(vals1, vals2)
        x[idx1] = min_vals
        x[idx2] = max_vals

    return x


class TestSystolicBubbleSort(unittest.TestCase):
    def test_sorted_output(self):
        for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
            original = torch.tensor([5, 1, 4, 2, 8], dtype=torch.float32)
            expected = torch.sort(original).values
            result = systolic_bubble_sort(original, device=device).cpu()
            self.assertTrue(torch.equal(result, expected), f"Failed on device {device}")


if __name__ == "__main__":
    unittest.main()
