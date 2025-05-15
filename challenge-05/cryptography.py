import unittest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os


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
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("Key must be bytes or bytearray")
        if len(key) not in (16, 24, 32):
            raise ValueError(f"Key must be 16, 24, or 32 bytes long. Got {len(key)}.")
        self.key = key
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
        self.key = b"0123456789abcdef"  # 16 bytes
        self.cipher = AESCipher(self.key)
        self.data = "This is some sensitive data to encrypt."

    def test_encrypt_decrypt(self):
        """Tests encryption and decryption."""
        encrypted_data = self.cipher.encrypt(self.data)
        decrypted_data = self.cipher.decrypt(encrypted_data)
        self.assertEqual(decrypted_data, self.data)

    def test_encrypt_decrypt_empty_string(self):
        """Tests encryption and decryption with an empty string."""
        encrypted_data = self.cipher.encrypt("")
        decrypted_data = self.cipher.decrypt(encrypted_data)
        self.assertEqual(decrypted_data, "")

    def test_invalid_key_length(self):
        """Tests that an error is raised for an invalid key length."""
        with self.assertRaises(ValueError):
            AESCipher(b"invalidkey")  # 8 bytes


def main():
    """Main function to run the cryptography demo."""
    print("Running Cryptography program execution...")
    key = b"0123456789abcdef"  # 16 bytes
    cipher = AESCipher(key)
    data = "This is some sensitive data to encrypt."

    encrypted_data = cipher.encrypt(data)
    print("Encrypted Data:", encrypted_data)

    decrypted_data = cipher.decrypt(encrypted_data)
    print("Decrypted Data:", decrypted_data)


if __name__ == "__main__":
    unittest.main()
