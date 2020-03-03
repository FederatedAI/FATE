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


### Affine Homomorphic Encryption
Affine homomorphic encryption is another kind of addition homomorphic encryption.

#### keygen
First generate a big integer n, then generate another three big integer a, inv_a, b, (a, n) = 1, a * inv % n = 1.

#### Encrypt
1. E(x) = (a * x + b) % n, recorded as pair (E(x), 1), E(x) is ciphertext, 1 means the bias b' coefficient is 1. 

#### Addition and Scalar Multiplication
1. E(x + y) = (E(x), 1) + (E(y), 1) = ((a * x + b) + (a * y + b), 1 + 1) = ((a * (x + y) + 2 * b), 2) = (E(x + y), 2)
2. scalar * E(x) = Scalar * (E(x), 1) = (E(scalar * x), scalar * 1)

#### Decrypt
Decrypt (E(x), k): remember that (E(x), k) = a * x + k * b, Dec((E(x), k) = inv_a * (E(x) - k * b) % n = inv_a * (a * x) % n = x % n


### IterativeAffine Homomorphic Encryption
Iterative Affine homomorphic encryption is another kind of addition homomorphic encryption.

#### keygen
Generate an key-tuple array, the element in the array is a tuple (a, inv_a, n), where (a, n) = 1, inv_a * a % n = 1. The array is sorted by n. 

#### Encrypt
1. E(x) = Enc_n o ... o Enc_1(x), Enc_r(x) = a_r * x % n_r. a_r, n_r is the r-th element of key-tuple array.

#### Addition and Scalar Multiplication
2. E(x + y) = E(x) + E(y) = Enc_n o ... o Enc_1(x + y)
3. scalar * E(x) = scalar * (Enc_n o ... o Enc_1(x)) = Enc_n o ... o Enc_1(scalar * x)

#### Decrypt

Dec(E(x)) = Dec_1 o ... o Dec_n(x), Dec_r(x) = (inv_a)_r * (a_r * x) % n_r = x % n_r


### Diffne Hellman Key Exchange

Diffie–Hellman key exchange is a method of securely exchanging cryptographic keys over a public channel 

#### protocol
1. keygen: generate big prime number p and g, where g is a primitive root modulo p
2. Alice generate random number r1;
Bob generate random number r2.
3. Alice calculate g^{r1} (mod p) then send to Bob; 
Bob calculate g^{r2} (mod p) then send to Alice.
4. Alice calculate (g^{r2} (mod p))^{r1} (mod p) = g^{r1 r2} (mod p);
Bob calculate (g^{r1} (mod p))^{r2} (mod p) = g^{r1 r2} (mod p);

g^{r1 r2} (mod p) is the share key.

#### How to use

```python
>>> from federatedml.secureprotol.diffie_hellman import DiffieHellman
>>> p, g = DiffieHellman.key_pair()
>>> import random
>>> r1 = random.randint(1, 10000000)
>>> r2 = random.randint(1, 10000000)
>>> key1 = DiffieHellman.decrypt(DiffieHellman.encrypt(g, r1, p), r2, p)
>>> key2 = DiffieHellman.decrypt(DiffieHellman.encrypt(g, r2, p), r1, p)
>>> assert key1 == key2
```

### RSA encryption
This encryption method generates three very large positive integers e, d and n. Let e, n as the public-key and d as the privacy-key. While giving data v, the encrypt operator will do en_v = v ^ e （mod n）， and the decrypt operator will do de_v = en_v ^ d (mod n)


### Fake encryption:
 It will do nothing and return input data during encryption and decryption.

## Encode
Encode module provide some method including "md5", "sha1", "sha224", "sha256", "sha384", "sha512" for data encoding. This module can help you to encode your data more convenient. It also supports for adding salt in front of data or behind data. For the encoding result, you can choose transform it to base64 or not.

## SecretShare MPC Protocol
### SPDZ

SPDZ([Ivan Damg˚ard](https://eprint.iacr.org/2011/535.pdf), [Marcel Keller](https://eprint.iacr.org/2017/1230.pdf)) is a multiparty computation scheme based on somewhat homomorphic encryption (SHE). 

#### init
```python
from arch.api import session
from arch.api import federation
s = session.init("session_name", 0)
federation.init("session_name", {
    "local": {
        "role": "guest",
        "party_id": 1000
    },
    "role": {
        "host": [999],
        "guest": [1000]
    }
 })
partys = federation.all_parties()
# [Party(role=guest, party_id=1000), Party(role=host, party_id=999)]
```
#### spdz env
tensor should be created and processed in spdz env:
```python
from federatedml.secureprotol.spdz import SPDZ
with SPDZ() as spdz:
    ...
```
#### create tenser
We currently provide two implementations of fixed point tensor:

1. one is based on numpy's array for non-distributed use:
    ```python
    from federatedml.secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor
    ```
    - on guest side(assuming local Party is partys[0]): 
    ```python
        import numpy as np
        data = np.array([[1,2,3], [4,5,6]])
        with SPDZ() as spdz:
            x = FixedPointTensor.from_source("x", data)
            y = FixedPointTensor.from_source("y", partys[1])
        ```

    - on host side(assuming PartyId is partys[1]):
    ```python
        import numpy as np
        data = np.array([[3,2,1], [6,5,4]])
        with SPDZ() as spdz:
            y = FixedPointTensor.from_source("y", data)
            x = FixedPointTensor.from_source("x", partys[1])
    ```

2. one based on a table for distributed use:
    ```python
    from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor
    ```
    - on guest side(assuming PartyId is partys[0]): 
    ```python
        data = session.parallelize(np.array([1,2,3]), np.array([4,5,6]))
        with SPDZ() as spdz:
            x = FixedPointTensor.from_source("x", data)
            y = FixedPointTensor.from_source("y", party_1)
        ```

    - on host side(assuming PartyId is partys[1]):
    ```python
        data = session.parallelize(np.array([3,2,1]), np.array([6,5,4]))
        with SPDZ() as spdz:
            y = FixedPointTensor.from_source("y", data)
            x = FixedPointTensor.from_source("x", party_0)
    ```

When tensor created from a provided data, data is split into n shares and every party gets a different one. 
#### rescontruct
Value can be rescontructed from tensor

```python
x.get() # array([[1, 2, 3],[4, 5, 6]])
y.get() # array([[3, 2, 1],[6, 5, 4]])
```

#### add/minus
You can add or subtract tensors

```python
z = x + y
t = x - y
```
#### dot
You can do dot arithmetic:
```python
x.dot(y)
```

#### einsum (numpy version only)
When using numpy's tensor, powerful einsum arithmetic is available:
```python
x.einsum(y, "ij,kj->ik")  # dot
```

