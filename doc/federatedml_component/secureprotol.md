# Secure Protocol

## Encrypt

Encrypt module provides some encryption methods for data. It contains
[Paillier](https://en.wikipedia.org/wiki/Paillier_cryptosystem),
[RSA](https://en.wikipedia.org/wiki/RSA_\(cryptosystem\))

### Paillier encryption

Paillier encryption is a kind of addition homomorphic encryption which
belongs to probabilistic asymmetric algorithm.


  - Generate key pairs  
  
    To generate key pairs for Paillier, two big prime numbers are
    needed. The generator will randomly pick two prime numbers, \(p\)
    and \(q\), whose bit length has been set. Then, p and q will be used
    as private keys to generate PrivateKey object.
    
    Then use \(n = pq\) as public key to generate PublicKey object.

  - Encrypt by PublicKey  
    
    Encryption of Paillier follows the steps below.

    1.  Encode: Paillier algorithm is applicable only to integer. Thus the
        input number will be encoded as a integer.
    
    2.  Encrypt: The principle of encryption can be referred to
        [here](https://en.wikipedia.org/wiki/Paillier_cryptosystem) or
        [this paper](http://www.cs.tau.ac.il/~fiat/crypt07/papers/Pai99pai.pdf)
    
    3.  Apply Obfuscator: Apply an obfuscator so that every encrypted
        number is different even if the plaintext is the same.

  - Decrypt by PrivateKey

    1.  Decrypt: Same principle introduction with encrypt above.
    
    2.  Decode: decode back as what it is.

  - Addition and Scalar Multiplication  
    
    Please refer to the links above with encrypt step for details.

### RSA encryption

This encryption method generates three very large positive integers
$e$, $d$ and $n$. Let $e$, $n$ as the public-key and $d$ as the
privacy-key. While giving data $v$, the encrypt operator will do
$$ enc(v) = v^e \pmod{n}, $$
and the decrypt operator will do
$$ dec(v) = enc(v) ^ d \pmod{n} $$

## Hash Factory

Hash factory provides following data encoding methods: "md5", "sha1", "sha224",
"sha256", "sha384", "sha512", "sm3". This module is meant to make hashing operation with more convenient. 
It also supports adding salt and outputing results to base64 format.

## Diffne Hellman Key Exchange

Diffie–Hellman key exchange is a method to exchange cryptographic keys
over a public channel securely

### Protocol

1.  keygen: generate big prime number $p$ and $g$, where $g$ is a
    primitive root modulo $p$
2.  Alice generates random number $r_1$; Bob generates random number
    $r_2$.
3.  Alice calculates $g^{r_1}\pmod{p}$ then send to Bob; Bob
    calculates $g^{r_2}\pmod{p}$ then sends to Alice.
4.  Alice calculates
    $(g^{r_2})^{r_1}\pmod{p}) = g^{r_1 r_2} \pmod{p}$; Bob calculates
    $(g^{r_1})^{r_2}\pmod{p} = g^{r_1 r_2} \pmod{p}$;
    $g^{r_1 r_2}\pmod{p}$ is the share key.

### How to use

```python
from federatedml.secureprotol.diffie_hellman import DiffieHellman
p, g = DiffieHellman.key_pair()
import random
r1 = random.randint(1, 10000000)
r2 = random.randint(1, 10000000)
key1 = DiffieHellman.decrypt(DiffieHellman.encrypt(g, r1, p), r2, p)
key2 = DiffieHellman.decrypt(DiffieHellman.encrypt(g, r2, p), r1, p)
assert key1 == key2
```

## SecretShare MPC Protocol(SPDZ)

SPDZ([Ivan Damg˚ard](https://eprint.iacr.org/2011/535.pdf),
[Marcel Keller](https://eprint.iacr.org/2017/1230.pdf)) is a
multiparty computation scheme based on somewhat homomorphic encryption
(SHE).

### How To Use

  - init
  
    ```python
    from fate_arch.session import Session
    s = Session()
    
    # on guest side
    guest_party_id = 10000
    host_party_id = 10001
    guest_proxy_ip = "192.168.0.2"  # Generally, it is your current machine IP
    federation_id = "spdz_demo"     # choose a common federation id (this should be same in both site)
    session_id = "_".join([federation_id, "guest", str(guest_party_id)])
    s.init_computing(session_id)
    s.init_federation(federation_id,
                      runtime_conf={
                          "local": {"role": "guest", "party_id": guest_party_id},
                          "role": {"guest": [guest_party_id], "host": [host_party_id]},
                      },
                      service_conf={"host": guest_proxy_ip, "port": 9370})
    s.as_global()
    partys = s.parties.all_parties
    # [Party(role=guest, party_id=10000), Party(role=host, party_id=10001)]

    # on host side
    guest_party_id = 10000
    host_party_id = 10001
    host_proxy_ip = "192.168.0.1"   # Generally, it is your current machine IP
    federation_id = "spdz_demo"     # choose a common federation id (this should be same in both site)
    session_id = "_".join([federation_id, "host", str(host_party_id)])
    s.init_computing(session_id)
    s.init_federation(federation_id,
                      runtime_conf={
                          "local": {"role": "host", "party_id": host_party_id},
                          "role": {"guest": [guest_party_id], "host": [host_party_id]},
                      },
                      service_conf={"host": host_proxy_ip, "port": 9370})
    s.as_global()
    partys = s.parties.all_parties
    # [Party(role=guest, party_id=10000), Party(role=host, party_id=10001)]
    ```

  - spdz env  
    tensor should be created and processed in spdz env:

    ```python
    from federatedml.secureprotol.spdz import SPDZ
    with SPDZ() as spdz:
        ...
    ```

  - create tenser  
    We currently provide two implementations of fixed point tensor:

    1.  one is based on numpy's array for non-distributed
    
        ```python
        from federatedml.secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor
        
        # on guest side(assuming local Party is partys[0]): 
        import numpy as np
        data = np.array([[1,2,3], [4,5,6]])
        with SPDZ() as spdz:
            x = FixedPointTensor.from_source("x", data)
            y = FixedPointTensor.from_source("y", partys[1])
        
        # on host side(assuming PartyId is partys[1]):
        import numpy as np
        data = np.array([[3,2,1], [6,5,4]])
        with SPDZ() as spdz:
            y = FixedPointTensor.from_source("y", data)
            x = FixedPointTensor.from_source("x", partys[0])
        ```
    
    2.  one based on a table for distributed
        
        ```python
        from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor
        
        # on guest side(assuming PartyId is partys[0]): 
        data = s.computing.parallelize([np.array([1,2,3]), np.array([4,5,6])], include_key=False, partition=2)
        with SPDZ() as spdz:
            x = FixedPointTensor.from_source("x", data)
            y = FixedPointTensor.from_source("y", party[1])
        
        # on host side(assuming PartyId is partys[1]):
        data = session.parallelize([np.array([3,2,1]), np.array([6,5,4])], include_key=False, partition=2)
        with SPDZ() as spdz:
            y = FixedPointTensor.from_source("y", data)
            x = FixedPointTensor.from_source("x", party[0])
        ```
      
      When tensor is created from a provided data, data is split into n shares and every party gets a different one.

  - rescontruct  
    
    Value can be rescontructed from tensor

    ```python
    x.get() # array([[1, 2, 3],[4, 5, 6]])
    y.get() # array([[3, 2, 1],[6, 5, 4]])
    ```

  - add/minus  
    
    You can add or subtract tensors

    ```python
    z = x + y
    t = x - y
    ```

  - dot  

    You can do dot arithmetic:
    
    ```python
    x.dot(y)
    ```

  - einsum (numpy version only)  
    
    When using numpy's tensor, powerful einsum arithmetic is available:

    ```python
    x.einsum(y, "ij,kj->ik")  # dot
    ```

## Oblivious Transfer

FATE implements Oblivious Transfer(OT) protocol based on work by Eduard
Hauck and Julian Loss. For more information, please refer
[here](https://eprint.iacr.org/2017/1011).

## Feldman Verifiable secret sharing

Feldman Verifiable secret sharing
[VSS](https://www.cs.umd.edu/~gasarch/TOPICS/secretsharing/feldmanVSS.pdf)
is an information-theoretic secure method to share secrets between
multi-parties.

### Protocol

1.  System parameters
    
    1.  1024-bits prime number $p$ and $g$ , 160-bits prime-order subgroup: $q$
    
    2.  Set share\_amount, it is the number of pieces the secret will be split into.

2.  Encrypt
    
    1.  Generate $k-1$ random numbers, which is ${a_0, a_1, a_2, ... ,a_{k-1}}$, denotes a polynomial of
        degree $k-1$, which is shown as
        $f(x)=a_0+a_1x+a_2x^2+...+a_{k-1}x^{k-1}$. where $a_0$ is
        the secret number, which requires a number of $k$ points to calculate.
    
    2.  Take $k$ points on the polynomial, generate $k$ sub-keys, such as $\{\langle1, f(1)\rangle, \langle 2,f(2)\rangle,\dots, \langle k, f(k)\rangle\}$
    
    3.  Generate commitments $c_i$ according to the $k$ coefficients, $c_i=g^{a_i}$

3.  Sub-key holder performs validation: $g^y=c_0c_1c_2c_3...c_{k-1}$,
    verifies that the sub-key is valid.

4.  Using Lagrange interpolation to recover secret.

### How to use

```python
from federatedml.secureprotol.secret_sharing.verifiable_secret_sharing.feldman_verifiable_secret_sharing import FeldmanVerifiableSecretSharing
vss = FeldmanVerifiableSecretSharing()
vss.key_pair()
vss.set_share_amount(3)
s = -5.98
sub_key, commitment = vss.encrypt(s) # generate sub-key and commitment
vss.verify(sub_key[i], commitment) # return True or False
x, y = zip(*sub_key)
secret = vss.decrypt(x,y) # assert s == secret
```
