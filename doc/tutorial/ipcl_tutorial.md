# IPCL Tutorial

## Intel Paillier Cryptosystem Library

[Intel Paillier Cryptosystem Library](https://github.com/intel/pailliercryptolib) (IPCL) is an open-source library which provides accelerated performance of a partial homomorphic encryption (HE), named Paillier cryptosystem, by utilizing Intel® [Integrated Performance Primitives Cryptography](https://github.com/intel/ipp-crypto) technologies on Intel CPUs supporting the AVX512IFMA instructions. The library is written in modern standard C++ and provides the essential API for the Paillier cryptosystem scheme. Intel Paillier Cryptosystem Library is certified for ISO compliance.

## Enable IPCL as Paillier Scheme in FATE

[Intel Paillier Cryptosystem Library - Python](https://github.com/intel/pailliercryptolib_python) (`ipcl_python`), a Python extension package of IPCL, is enabled in FATE as an optional Paillier scheme, alongside original FATE Paillier scheme.

The following sections introduce:

- How to enable IPCL when building FATE
- How to use IPCL as Paillier scheme when performing privacy-preseving machine learning tasks
- How IPCL is integrated into FATE and corresponding code changes

### Build FATE with IPCL

- **Build Docker Images**
  
  To build FATE images with `ipcl_python` enabled, please refer to [README](https://github.com/FederatedAI/FATE-Builder/blob/main/docker-build/README.md#environments) of [FATE-Builder](https://github.com/FederatedAI/FATE-Builder). Set `Build_IPCL=1`.

### IPCL Usage

- In machine learning task
  
  Set the value of `encrypt_param.method` to "ipcl" when configuring the model in `federatedml.components`.

- In `fate_test`
  
  - Install `fate_test` according to [`fate_test_tutorial`](https://github.com/FederatedAI/FATE/blob/master/doc/tutorial/fate_test_tutorial.md)

  - Test Paillier operations with both original FATE Paillier scheme and IPCL

    ```bash
    fate_test op-test paillier
    ```

### Integration Steps

The following steps briefly introduce how IPCL is integrated into FATE, which have been already done since FATE v1.10.0.

- Add IPCL as an encryption parameter `consts.PAILLIER_IPCL`

  - `python/federatedml/util/consts.py`

    ```diff
      PAILLIER = 'Paillier'
    + PAILLIER_IPCL = 'IPCL'
    ```

  - `python/federatedml/param/encrypt_param.py`

    ```diff
    + elif user_input == "ipcl":
    +     self.method = consts.PAILLIER_IPCL
    ```

- Add `ipcl_python` as the backend of secure protocol

  - `python/federatedml/secureprotocol/encrypt.py`

    ```diff
    + from ipcl_python import PaillierKeypair as IpclPaillierKeypair
    + from ipcl_python import PaillierEncryptedNumber as IpclPaillierEncryptedNumber
      ...
    + class IpclPaillierEncrypt(Encrypt):
    + ...
    ```

  - `python/federatedml/secureprotocol/__init__.py`

    ```diff
    - from federatedml.secureprotol.encrypt import RsaEncrypt, PaillierEncrypt
    + from federatedml.secureprotol.encrypt import RsaEncrypt, PaillierEncrypt, IpclPaillierEncrypt
      from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator

    - __all__ = ['RsaEncrypt', 'PaillierEncrypt', 'EncryptModeCalculator']
    + __all__ = ['RsaEncrypt', 'PaillierEncrypt', 'IpclPaillierEncrypt', 'EncryptModeCalculator']
    ```

- Add `IpclPaillierEncrypt` as the model's `cipher_operator`, e.g., `hetero_logistic_regression`

  - `python/federatedml/linear_model/coordinated_linear_model/logistic_regression/hetero_logistic_regression/hetero_lr_base.py`

    ```diff
    - from federatedml.secureprotol import PaillierEncrypt
    + from federatedml.secureprotol import PaillierEncrypt, IpclPaillierEncrypt
      ...
    -         self.cipher_operator = PaillierEncrypt()
    +
    +         if params.encrypt_param.method == consts.PAILLIER:
    +             self.cipher_operator = PaillierEncrypt()
    +         elif params.encrypt_param.method == consts.PAILLIER_IPCL:
    +             self.cipher_operator = IpclPaillierEncrypt()
    +         else:
    +             raise ValueError(f"Unsupported encryption method: {params.encrypt_param.method}")
    +
    ```

## Current Status

For now, `hetero_logistic_regression` and `hetero_secure_boost` models are supported by IPCL.
