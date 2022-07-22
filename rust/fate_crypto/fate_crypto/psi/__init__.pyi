from typing import overload

class Curve25519(object):
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, key): ...
    def get_private_key(self) -> bytes: ...
    def encrypt(self, m: bytes) -> bytes:
        """encrypt message.
        This contains flowing steps:
        1. Perform hashing to the group using the Elligator2 map to
            generate a Montgomery Point `A` in curve25519.
            (See https://tools.ietf.org/html/draft-irtf-cfrg-hash-to-curve-10#section-6.7.1)
        2. Perform scalar multipication `A * Scalar(secret)` to generate Montgomery Point `B`
        3. return `B` represents by 32 bytes value
        Args:
            m (bytes): message to encrypt

        Returns:
            bytes: encryptd message in 32-length bytes
        """
        ...
    def diffie_hellman(self, pub: bytes) -> bytes:
        """generate diffie_hellman like sharedsecret.

        Args:
            pub (bytes): encryted message in 32-length bytes.

        Returns:
            bytes: sharedsecret in 32-length bytes
        """
        ...
