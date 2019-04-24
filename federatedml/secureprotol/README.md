### Encrypt
Encrypt module provides some encryption method for data. It contains [Paillier](https://en.wikipedia.org/wiki/Paillier_cryptosystem), [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)), and Fake method.
1. Paillier encryption: It supports the encryption method using Paillier, which  is a probabilistic asymmetric algorithm for public key cryptography.
2. RSA encryption: This encryption method generates three very large positive integers e, d and n. Let e, n as the public-key and d as the privacy-key. While giving data v, the encrypt operator will do en_v = v ^ e （mod n）， and the decrypt operator will do de_v = en_v ^ d (mod n)
3. Fake encryption: It will do nothing and return input data during encryption and decryption.

### Encode
Encode module provide some method including "md5", "sha1", "sha224", "sha256", "sha384", "sha512" for data encoding. This module can help you to encode your data more convenient. It also supports for adding salt in front of data or behind data. For the encoding result, you can choose transform it to base64 or not.