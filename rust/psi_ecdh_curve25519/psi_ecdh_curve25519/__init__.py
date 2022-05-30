import random
from typing import Optional
from .psi_ecdh_curve25519 import Secret

__all__ = ["ECDHCurve25519"]


class ECDHCurve25519(object):
    def __init__(self, private_key: Optional[bytes] = None) -> None:
        if private_key is None:
            private_key = (
                random.SystemRandom().getrandbits(32 * 8).to_bytes(32, "little")
            )
        else:
            if not isinstance(private_key, bytes) or len(private_key) != 32:
                raise KeyError(f"private key should be 32 length bytes")

        self._private = private_key

    def get_private_key(self) -> bytes:
        return self._private

    def __getstate__(self):
        return self._private

    def __setstate__(self, private):
        self._private = private

    @property
    def _inside(self):
        if not hasattr(self, "_secret"):
            setattr(self, "_secret", Secret(self._private))
        return getattr(self, "_secret")

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
        return self._inside.encrypt(m)

    def diffie_hellman(self, pub: bytes) -> bytes:
        """generate diffie_hellman like sharedsecret.

        Args:
            pub (bytes): encryted message in 32-length bytes.

        Returns:
            bytes: sharedsecret in 32-length bytes
        """
        return self._inside.diffie_hellman(pub)
