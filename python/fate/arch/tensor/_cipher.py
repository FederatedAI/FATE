from .paillier import keygen as paillier_keygen


def keygen(kind, options):
    return paillier_keygen(**options)
