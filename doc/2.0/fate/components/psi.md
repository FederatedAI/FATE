# PSI

This module implements PSI(Private Set Intersection)
based on [elliptic curve Diffie-Hellman scheme](https://en.wikipedia.org/wiki/Elliptic-curve_Diffie–Hellman).
ECDH mode currently uses [Curve25519](https://en.wikipedia.org/wiki/Curve25519),  
which offers 128 bits of security with key size of 256 bits.

Below is an illustration of ECDH intersection.

![Figure 1 (ECDH
PSI)](../../images/ecdh_intersection.png)

For details on how to hash value to given curve,
please refer [here](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-hash-to-curve-10#section-6.7.1).

Note that starting in ver 2.0.0-beta, data uploaded should always have sample id and match id;
for data sets that originally only contain id, please specify 'extend_sid' in upload config
as in this [example](../../../examples/pipeline/upload/test_upload.py).