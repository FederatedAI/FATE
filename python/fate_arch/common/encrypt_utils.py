import base64

from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP as OAEP_cipher
# update RSA OAEP for encryption.It is described in RFC8017 where it is called RSAES-OAEP.


def rsa_key_generate():
    random_generator = Random.new().read
    rsa = RSA.generate(2048, random_generator)
    private_pem = rsa.exportKey().decode()
    public_pem = rsa.publickey().exportKey().decode()
    with open('private_key.pem', "w") as f:
        f.write(private_pem)
    with open('public_key.pem', "w") as f:
        f.write(public_pem)
    return private_pem, public_pem


def encrypt_data(public_key, msg):
    cipher = OAEP_cipher.new(RSA.importKey(public_key))
    encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf8"))))
    return encrypt_text.decode('utf-8')


def pwdecrypt(private_key, encrypt_msg):
    cipher = OAEP_cipher.new(RSA.importKey(private_key))
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg))
    return back_text.decode('utf-8')


def test_encrypt_decrypt():
    msg = "fate"
    private_key, public_key = rsa_key_generate()
    encrypt_text = encrypt_data(public_key, msg)
    print(encrypt_text)
    decrypt_text = pwdecrypt(private_key, encrypt_text)
    print(msg == decrypt_text)
