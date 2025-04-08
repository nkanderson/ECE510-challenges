import unittest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
import cProfile
import pstats
import io


class AESCipher:
    """
    A class for encrypting and decrypting data using AES.
    """

    def __init__(self, key):
        """
        Initializes the cipher with a key.

        Args:
            key: The encryption key (must be 16, 24, or 32 bytes long).
        """
        self.key = key
        if len(key) not in (16, 24, 32):
            raise ValueError("Key must be 16, 24, or 32 bytes long")
        self.bs = AES.block_size  # 16

    def encrypt(self, raw):
        """
        Encrypts the raw data.

        Args:
            raw: The raw data to be encrypted (string).

        Returns:
            The encrypted data (bytes).
        """
        raw = pad(raw.encode("utf-8"), self.bs)
        iv = os.urandom(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        """
        Decrypts the encrypted data.

        Args:
            enc: The encrypted data (bytes).

        Returns:
            The decrypted data (string).
        """
        iv = enc[: AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        raw = cipher.decrypt(enc[AES.block_size :])
        return unpad(raw, self.bs).decode("utf-8")


class TestCryptography(unittest.TestCase):
    def setUp(self):
        self.key = b"This is a secret key123456789"  # 32 bytes
        self.cipher = AESCipher(self.key)
        self.data = "This is some sensitive data to encrypt."

    def test_encrypt_decrypt(self):
        """Tests encryption and decryption."""
        encrypted_data = self.cipher.encrypt(self.data)
        decrypted_data = self.cipher.decrypt(encrypted_data)
        self.assertEqual(decrypted_data, self.data)

    def test_encrypt_decrypt_empty_string(self):
        """Tests encryption and decryption with an empty string."""
        empty_data = ""
        encrypted_data = self.cipher.encrypt(empty_data)
        decrypted_data = self.cipher.decrypt(encrypted_data)
        self.assertEqual(decrypted_data, empty_data)

    def test_invalid_key_length(self):
        """Tests that an error is raised for an invalid key length."""
        with self.assertRaises(ValueError):
            AESCipher(b"invalidkey")  # 8 bytes


def main():
    """Main function to run the cryptography program and tests."""
    print("Running Cryptography program execution...")
    # Example
    key = b"This is a secret key123456789"  # 32 bytes
    cipher = AESCipher(key)
    data = "This is some sensitive data to encrypt."
    encrypted_data = cipher.encrypt(data)
    print("Encrypted Data:", encrypted_data)
    decrypted_data = cipher.decrypt(encrypted_data)
    print("Decrypted Data:", decrypted_data)

    # Profile the execution
    profiler = cProfile.Profile()
    profiler.enable()
    cipher.encrypt(data)  # Profile the encryption
    cipher.decrypt(encrypted_data)  # also profile decryption
    profiler.disable()

    # Save the profiling results to a file
    profiler.dump_stats("cryptography.prof")

    # Print the statistics to the console
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats()


if __name__ == "__main__":
    unittest.main()
