import phe
import time
import fate_tensor
import numpy as np

shape = 1000
data = np.random.random((shape, 1))


def bench(name, func, repeat_num):
    t = time.time()
    for _ in range(repeat_num):
        func()
    cost = (time.time() - t) / float(repeat_num * shape)
    print(f"{name}: {1 / cost} each secends, {cost} each run")
    return cost


pubkey, prikey = phe.generate_paillier_keypair(n_length=1024)
phe_plaintext = data.reshape((shape,)).tolist()
phe_ciphertext = [pubkey.encrypt(x) for x in phe_plaintext]


def encrypt():
    for i in range(shape):
        pubkey.encrypt(phe_plaintext[i])


def decrypt():
    for i in range(shape):
        prikey.decrypt(phe_ciphertext[i])


def add():
    for i in range(shape):
        phe_ciphertext + phe_ciphertext


repeat_num = 10
bench("phe_encrypt", encrypt, repeat_num)
bench("phe_decrypt", decrypt, repeat_num)
bench("phe_add", decrypt, repeat_num)

repeat_num = 10
pk, ek = fate_tensor.keygen(1024)
rust_plaintext = data
rust_ciphertext = pk.encrypt_f64(data)
bench("rust_encrypt", lambda: pk.encrypt_f64(rust_plaintext), repeat_num)
bench("rust_decrypt", lambda: ek.decrypt_f64(rust_ciphertext), repeat_num)
bench("rust_add", lambda: rust_ciphertext.add_cipherblock(rust_ciphertext), repeat_num)

repeat_num = 10
pk, ek = fate_tensor.keygen(1024)
rust_plaintext = data
rust_ciphertext = pk.encrypt_f64_par(data)
bench("rust_par_encrypt", lambda: pk.encrypt_f64_par(rust_plaintext), repeat_num)
bench("rust_par_decrypt", lambda: ek.decrypt_f64_par(rust_ciphertext), repeat_num)
bench(
    "rust_par_add",
    lambda: rust_ciphertext.add_cipherblock_par(rust_ciphertext),
    repeat_num,
)
