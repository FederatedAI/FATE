def phe_keygen(kind, options):
    if kind == "paillier":
        from .paillier import PaillierCipher

        return PaillierCipher.keygen(**options)
    if kind == "paillier_vector_based":
        from .paillier_vertor_based import PaillierCipher

        return PaillierCipher.keygen(**options)
    elif kind == "mock":
        from .mock import PaillierCipher

        return PaillierCipher(**options)
    else:
        raise ValueError(f"Unknown PHE keygen kind: {kind}")
