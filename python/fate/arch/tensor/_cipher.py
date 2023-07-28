from .mock import keygen as mock_keygen


def phe_keygen(kind, options):
    if kind == "paillier_vector_based":
        from .paillier_vertor_based import PaillierCipher

        return PaillierCipher.keygen(**options)
    elif kind == "mock":
        return mock_keygen(**options)
    else:
        raise ValueError(f"Unknown PHE keygen kind: {kind}")
