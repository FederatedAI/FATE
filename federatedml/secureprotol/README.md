## Encrypt
Encrypt module provides some encryption method for data. It contains [Paillier](https://en.wikipedia.org/wiki/Paillier_cryptosystem), [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)), and Fake method.

### Paillier encryption
Paillier encryption is a kind of addition homomorphic encryption who belongs to probabilistic asymmetric algorithm.

#### Generate key pairs
To generate key pairs for Paillier, two big prime numbers are needed. The generator will randomly pick two prime numbers, p and q, whose bit length has been set. Then, p and q will be used as private keys to generate PrivateKey object.

Then use n = p * q as public key to generate PublicKey object.

#### Encrypt by PublicKey
Encryption of Paillier follows the steps below.
1. Encode: Paillier algorithm is applicable only to integer. Thus the input number will be encoded as a integer.

2. Encrypt: The principle of encryption can be referred to [here](https://en.wikipedia.org/wiki/Paillier_cryptosystem) or [this paper](http://www.cs.tau.ac.il/~fiat/crypt07/papers/Pai99pai.pdf)

3. Apply Obfuscator: Apply an obfuscator so that every encrypted number is different even if the plaintext is same.

#### Decrypt by PrivateKey
1. Decrypt: Same principle introduction with encrypt above.

2. Decode: decode back as what it is.

#### Addition and Scalar Multiplication
Please refer to the same pages with encrypt step above for details.

### RSA encryption
This encryption method generates three very large positive integers e, d and n. Let e, n as the public-key and d as the privacy-key. While giving data v, the encrypt operator will do en_v = v ^ e （mod n）， and the decrypt operator will do de_v = en_v ^ d (mod n)


### Fake encryption:
 It will do nothing and return input data during encryption and decryption.

## Encode
Encode module provide some method including "md5", "sha1", "sha224", "sha256", "sha384", "sha512" for data encoding. This module can help you to encode your data more convenient. It also supports for adding salt in front of data or behind data. For the encoding result, you can choose transform it to base64 or not.