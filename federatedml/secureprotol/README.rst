Encrypt
=======

Encrypt module provides some encryption methods for data. It contains `[Paillier] <https://en.wikipedia.org/wiki/Paillier_cryptosystem>`_, `[RSA] <https://en.wikipedia.org/wiki/RSA_(cryptosystem)>`_, and Fake method.

Paillier encryption
-------------------

Paillier encryption is a kind of addition homomorphic encryption which belongs to probabilistic asymmetric algorithm.

:Generate key pairs:
    To generate key pairs for Paillier, two big prime numbers are needed. The generator will randomly pick two prime numbers, :math:`p` and :math:`q`, whose bit length has been set. Then, p and q will be used as private keys to generate PrivateKey object.

    Then use :math:`n = pq` as public key to generate PublicKey object.

:Encrypt by PublicKey: Encryption of Paillier follows the steps below.

    1. Encode: Paillier algorithm is applicable only to integer. Thus the input number will be encoded as a integer.

    2. Encrypt: The principle of encryption can be referred to `[here] <https://en.wikipedia.org/wiki/Paillier_cryptosystem>`_ or `[this paper] <http://www.cs.tau.ac.il/~fiat/crypt07/papers/Pai99pai.pdf>`_

    3. Apply Obfuscator: Apply an obfuscator so that every encrypted number is different even if the plaintext is the same.


:Decrypt by PrivateKey:

    1. Decrypt: Same principle introduction with encrypt above.

    2. Decode: decode back as what it is.


:Addition and Scalar Multiplication: Please refer to the links above with encrypt step for details.



Affine Homomorphic Encryption
-----------------------------

Affine homomorphic encryption is another kind of addition homomorphic encryption.


:keygen: First generate a big integer :math:`n`, then generate another three big integers :math:`a, a^{-1}, b, (a, n) = 1, a * a^{-1}\equiv 1\pmod{n}`.


:Encrypt:

    :math:`E(x) = (a * x + b)\pmod{n}`, recorded as pair :math:`(E(x), 1)`, :math:`E(x)` is ciphertext, 1 means the bias :math:`b` is 1. 


:Addition and Scalar Multiplication:

    1. :math:`E(x + y) = (E(x), 1) + (E(y), 1) = ((a * x + b) + (a * y + b), 1 + 1) = ((a * (x + y) + 2 * b), 2) = (E(x + y), 2)`

    2. :math:`scalar * E(x) = Scalar * (E(x), 1) = (E(scalar * x), scalar * 1)`


:Decrypt: Decrypt :math:`(E(x), k)`: remember that :math:`(E(x), k) = a * x + k * b, Dec((E(x), k) = a^{-1} * (E(x) - k * b) \pmod{n} = a^{-1} * (a * x) \pmod{n} = x \pmod{n}`


IterativeAffine Homomorphic Encryption
--------------------------------------

Iterative Affine homomorphic encryption is another kind of addition homomorphic encryption.

:keygen: Generate an key-tuple array, the element in the array is a tuple :math:`(a, a^{-1}, n)`, where :math:`(a, n) = 1`, :math:`a^{-1} * a \equiv 1\pmod{n}`. The array is sorted by :math:`n`. 


:Encrypt: 
    .. math::
        E(x) = (Enc_n \circ\dots\circ Enc_1)(x)

    where :math:`Enc_r(x) = a_r * x % n_r. a_r`, :math:`n_r` is the r-th element of key-tuple array.


:Addition and Scalar Multiplication:

    1. :math:`E(x + y) = E(x) + E(y) = (Enc_n \circ\dots\circ Enc_1)(x + y)`

    2. :math:`scalar * E(x) = scalar * ((Enc_n\circ\dots\circ Enc_1)(x)) = (Enc_n\circ\dots\circ Enc_1)(scalar * x)`


:Decrypt:
    .. math::
        Dec(E(x)) = (Dec_1 \circ Dec_2 \circ \dots \circ Dec_n)(x)

    where :math:`Dec_r(x) = a_r^{-1} * (a_r * x) % n_r = x % n_r`



RSA encryption
--------------

This encryption method generates three very large positive integers :math:`e`, :math:`d` and :math:`n`. Let :math:`e`, :math:`n` as the public-key and d as the privacy-key. While giving data :math:`v`, the encrypt operator will do :math:`enc(v) = v ^ e \pmod{n}`， and the decrypt operator will do :math:`dec(v) = env(v) ^ d \pmod{n}`


Fake encryption
---------------

It will do nothing and return input data during encryption and decryption.


Encode
======

Encode module provides some method including "md5", "sha1", "sha224", "sha256", "sha384", "sha512" for data encoding. This module can help you to encode your data with more convenient. It also supports for adding salt in front of data or behind data. For the encoding result, you can choose transform it to base64 or not.


Diffne Hellman Key Exchange
===========================

Diffie–Hellman key exchange is a method to exchange cryptographic keys over a public channel securely

Protocol
--------

1. keygen: generate big prime number :math:`p` and :math:`g`, where :math:`g` is a primitive root modulo :math:`p`

2. Alice generates random number :math:`r_1`; Bob generates random number :math:`r_2`.

3. Alice calculates :math:`g^{r_1}\pmod{p}` then send to Bob;  Bob calculates :math:`g^{r_2}\pmod{p}` then sends to Alice.

4. Alice calculates :math:`(g^{r_2})^{r_1}\pmod{p}) = g^{r_1 r_2} \pmod{p}`; Bob calculates :math:`(g^{r_1})^{r_2}\pmod{p} = g^{r_1 r_2} \pmod{p}`; :math:`g^{r_1 r_2}\pmod{p}` is the share key.


How to use
----------

.. code-block:: python

   from federatedml.secureprotol.diffie_hellman import DiffieHellman
   p, g = DiffieHellman.key_pair()
   import random
   r1 = random.randint(1, 10000000)
   r2 = random.randint(1, 10000000)
   key1 = DiffieHellman.decrypt(DiffieHellman.encrypt(g, r1, p), r2, p)
   key2 = DiffieHellman.decrypt(DiffieHellman.encrypt(g, r2, p), r1, p)
   assert key1 == key2



SecretShare MPC Protocol(SPDZ)
==============================

SPDZ(`[Ivan Damg˚ard] <https://eprint.iacr.org/2011/535.pdf>`_, `[Marcel Keller] <https://eprint.iacr.org/2017/1230.pdf>`_) is a multiparty computation scheme based on somewhat homomorphic encryption (SHE). 


How To Use
----------

:init:

    .. code-block:: python

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


:spdz env: tensor should be created and processed in spdz env:

    .. code-block:: python

        from federatedml.secureprotol.spdz import SPDZ
        with SPDZ() as spdz:
            ...


:create tenser: We currently provide two implementations of fixed point tensor:

    1. one is based on numpy's array for non-distributed use:
    
    .. code-block:: python
            
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
                x = FixedPointTensor.from_source("x", partys[1])


    2. one based on a table for distributed use:

    .. code-block:: python

       from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor
       
       # on guest side(assuming PartyId is partys[0]): 
       data = session.parallelize(np.array([1,2,3]), np.array([4,5,6]))
       with SPDZ() as spdz:
       x = FixedPointTensor.from_source("x", data)
       y = FixedPointTensor.from_source("y", party_1)
       
       # on host side(assuming PartyId is partys[1]):
       data = session.parallelize(np.array([3,2,1]), np.array([6,5,4]))
           with SPDZ() as spdz:
               y = FixedPointTensor.from_source("y", data)
               x = FixedPointTensor.from_source("x", party_0)


When tensor is created from a provided data, data is split into n shares and every party gets a different one.

:rescontruct: Value can be rescontructed from tensor

.. code-block:: python
   
   x.get() # array([[1, 2, 3],[4, 5, 6]])
   y.get() # array([[3, 2, 1],[6, 5, 4]])


:add/minus: You can add or subtract tensors

.. code-block:: python
   
   z = x + y
   t = x - y


:dot: You can do dot arithmetic:

.. code-block:: python

   x.dot(y)


:einsum (numpy version only): When using numpy's tensor, powerful einsum arithmetic is available:

.. code-block:: python
   
   x.einsum(y, "ij,kj->ik")  # dot
