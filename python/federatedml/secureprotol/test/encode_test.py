import unittest
from federatedml.secureprotol.encode import Encode


class TestEncode(unittest.TestCase):
    def test_compute(self):
        value_list = ["12345", "54321", "111111"]
        pre_salt = ""
        postfit_salt = "12345"

        sha256_base64_value_list = [
            "5KCpDlrAfVQ1xvJcTPfMVlvst5e7W4PFFbxCfvMqR3A=",
            "O90AvF51EJNqKGewUOtH4tUFyKM2y5NnlbUwRGb7kQw=",
            "M1RdblccBKI/fUivm9yrH3cCkK8lnmYYJPBe8E2FDWM="]

        sha256_value_list = [
            "e4a0a90e5ac07d5435c6f25c4cf7cc565becb797bb5b83c515bc427ef32a4770",
            "3bdd00bc5e7510936a2867b050eb47e2d505c8a336cb936795b5304466fb910c",
            "33545d6e571c04a23f7d48af9bdcab1f770290af259e661824f05ef04d850d63"]

        md5_value_list = ["8cfa2282b17de0a598c010f5f0109e7d",
                          "64bddc3ca51ad547e43f8e65cb5e2318",
                          "ff02c8a17f7fd875bd2e6d882fe7677d"]

        md5_base64_value_list = ["jPoigrF94KWYwBD18BCefQ==",
                                 "ZL3cPKUa1UfkP45ly14jGA==",
                                 "/wLIoX9/2HW9Lm2IL+dnfQ=="]

        sha1_base64_value_list = ["bur67wEzGYIqHzBAelNT93i1l5A=",
                                  "hhrM2zkxjD26v53sNtImywM5Q7k=",
                                  "X1M3cC7+mBF8IIHT6aghS/JJXvc="]

        sha1_value_list = ["6eeafaef013319822a1f30407a5353f778b59790",
                           "861accdb39318c3dbabf9dec36d226cb033943b9",
                           "5f5337702efe98117c2081d3e9a8214bf2495ef7"]

        sha224_base64_value_list = ["uVsQjSHFEq2gQLixymFYv5Ieht4p9v+MHr64kw==",
                                    "zAKzPY0k41ZbCyqTY4cBeNqOo7R7uca0f36Pjg==",
                                    "cNN9OVWrPqJ2g/Ve/w395o30Jxy7W3ol8NrF4w=="]

        sha224_value_list = ["b95b108d21c512ada040b8b1ca6158bf921e86de29f6ff8c1ebeb893",
                             "cc02b33d8d24e3565b0b2a9363870178da8ea3b47bb9c6b47f7e8f8e",
                             "70d37d3955ab3ea27683f55eff0dfde68df4271cbb5b7a25f0dac5e3"]

        sha512_base64_value_list = [
            "bnhO4jFWKBnxwBlowI2zlaF0cLRFlDFIBLI1jvRI94Ohq6KkvhZSm+HoC6bzEPaIFzjBzGx3kOZlLdnNlNJaVg==",
            "teHJ2xDTAHht99rky6eqMDaiINvTl30OzXq7sL9Dkk3NQ5GOJSd4ozempcTPgi8XD+uVtovhUPPkCOh1zJLLjg==",
            "LWE4K5RVIJ3p8RQ6JMuT/YYfIz/T5CPdx/8Z4+ywB7DOOS/7wFsR5pMrAvLb5u3G0auCf6d3lA4v69D7Vk98/Q=="]

        sha512_value_list = [
            "6e784ee231562819f1c01968c08db395a17470b44594314804b2358ef448f783a1aba2a4be16529be1e80ba6f310f6881738c1cc6c7790e6652dd9cd94d25a56",
            "b5e1c9db10d300786df7dae4cba7aa3036a220dbd3977d0ecd7abbb0bf43924dcd43918e252778a337a6a5c4cf822f170feb95b68be150f3e408e875cc92cb8e",
            "2d61382b9455209de9f1143a24cb93fd861f233fd3e423ddc7ff19e3ecb007b0ce392ffbc05b11e6932b02f2dbe6edc6d1ab827fa777940e2febd0fb564f7cfd"]

        sha384_base64_value_list = ["VQY6S0eKPD7KBYLKeBZ00Ys8Zr1HYUW4D9J82hmaDDBkNpoq9mvWchRo/isr/9Cb",
                                    "iF8kMFkBmrFbmUUrb0qs4j5ZWRXACfpKUUZr/4SqaI96EMJn7Atfk/z8JMoDyK4j",
                                    "wdy0BD5/rrkj2ABLchnCWMXGK6HKUc9NM23cepUdgEX9isxxEjynWfWkNwfObFpe"]

        sha384_value_list = [
            "55063a4b478a3c3eca0582ca781674d18b3c66bd476145b80fd27cda199a0c3064369a2af66bd6721468fe2b2bffd09b",
            "885f243059019ab15b99452b6f4aace23e595915c009fa4a51466bff84aa688f7a10c267ec0b5f93fcfc24ca03c8ae23",
            "c1dcb4043e7faeb923d8004b7219c258c5c62ba1ca51cf4d336ddc7a951d8045fd8acc71123ca759f5a43707ce6c5a5e"]

        ## test sha256, base64 = 1
        # encode_sha256_base64 = Encode("sha256", base64=1)
        # self.assertEqual(encode_sha256_base64.compute(value_list, pre_salt, postfit_salt)[0], sha256_base64_value_list)

        ## test sha256, base64 = 0
        # encode_sha256 = Encode("sha256", base64=0)
        # self.assertEqual(encode_sha256.compute(value_list, pre_salt, postfit_salt)[0], sha256_value_list)

        ## test md5, base64 = 1
        # encode_md5_base64 = Encode("md5", base64=1)

        # self.assertEqual(encode_md5_base64.compute(value_list, pre_salt, postfit_salt)[0], md5_base64_value_list)

        ## test md5, base64 = 0
        # encode_md5 = Encode("md5", base64=0)

        # self.assertEqual(encode_md5.compute(value_list, pre_salt, postfit_salt)[0], md5_value_list)

        ## test sha1, base64 = 1
        # encode_base64_sha1 = Encode("sha1", base64=1)

        # self.assertEqual(encode_base64_sha1.compute(value_list, pre_salt, postfit_salt)[0], sha1_base64_value_list)

        ## test sha1, base64 = 0
        # encode_sha1 = Encode("sha1", base64=0)
        # self.assertEqual(encode_sha1.compute(value_list, pre_salt, postfit_salt)[0], sha1_value_list)

        ## test sha1, base64 = 1
        # encode_base_sha224 = Encode("sha224", base64=1)
        # self.assertEqual(encode_base_sha224.compute(value_list, pre_salt, postfit_salt)[0], sha224_base64_value_list)

        ## test sha224, base64 = 0
        # encode_sha224 = Encode("sha224", base64=0)
        # self.assertEqual(encode_sha224.compute(value_list, pre_salt, postfit_salt)[0], sha224_value_list)

        ## test sha512, base64 = 1
        # encode_base_sha512 = Encode("sha512", base64=1)
        # self.assertEqual(encode_base_sha512.compute(value_list, pre_salt, postfit_salt)[0], sha512_base64_value_list)

        ## test sha224, base64 = 0
        # encode_sha512 = Encode("sha512", base64=0)
        # self.assertEqual(encode_sha512.compute(value_list, pre_salt, postfit_salt)[0], sha512_value_list)

        ## test sha384, base64 = 1
        # encode_base_sha384 = Encode("sha384", base64=1)
        # self.assertEqual(encode_base_sha384.compute(value_list, pre_salt, postfit_salt)[0], sha384_base64_value_list)

        ## test sha384, base64 = 0
        # encode_sha384 = Encode("sha384", base64=0)
        # self.assertEqual(encode_sha384.compute(value_list, pre_salt, postfit_salt)[0], sha384_value_list)

        #
        ## test id_map
        # encode_sha384 = Encode("sha384", base64=0)
        # sha384_value_list = [
        #    "55063a4b478a3c3eca0582ca781674d18b3c66bd476145b80fd27cda199a0c3064369a2af66bd6721468fe2b2bffd09b",
        #    "885f243059019ab15b99452b6f4aace23e595915c009fa4a51466bff84aa688f7a10c267ec0b5f93fcfc24ca03c8ae23",
        #    "c1dcb4043e7faeb923d8004b7219c258c5c62ba1ca51cf4d336ddc7a951d8045fd8acc71123ca759f5a43707ce6c5a5e"]
        # encode_value_list, id_map_pair = encode_sha384.compute(value_list, pre_salt, postfit_salt, id_map=True)
        # id_map_value_list = []
        # for i in range(len(encode_value_list)):
        #    id_map_value_list.append(id_map_pair[encode_value_list[i]])
        #
        # self.assertEqual(id_map_value_list, value_list)

        ########## test single value #####################
        value = value_list[0]

        # test sha256, base64 = 1
        encode_sha256_base64 = Encode("sha256", base64=1)
        sha256_base64_value = sha256_base64_value_list[0]
        self.assertEqual(encode_sha256_base64.compute(value, pre_salt, postfit_salt), sha256_base64_value)

        # test sha256, base64 = 0
        encode_sha256 = Encode("sha256", base64=0)
        sha256_value = sha256_value_list[0]
        self.assertEqual(encode_sha256.compute(value, pre_salt, postfit_salt), sha256_value)

        # test md5, base64 = 1
        encode_md5_base64 = Encode("md5", base64=1)
        md5_base64_value = md5_base64_value_list[0]

        self.assertEqual(encode_md5_base64.compute(value, pre_salt, postfit_salt), md5_base64_value)

        # test md5, base64 = 0
        encode_md5 = Encode("md5", base64=0)
        md5_value = md5_value_list[0]

        self.assertEqual(encode_md5.compute(value, pre_salt, postfit_salt), md5_value)

        # test sha1, base64 = 1
        encode_base64_sha1 = Encode("sha1", base64=1)
        sha1_base64_value = sha1_base64_value_list[0]
        self.assertEqual(encode_base64_sha1.compute(value, pre_salt, postfit_salt), sha1_base64_value)

        # test sha1, base64 = 0
        encode_sha1 = Encode("sha1", base64=0)
        sha1_value = sha1_value_list[0]
        self.assertEqual(encode_sha1.compute(value, pre_salt, postfit_salt), sha1_value)

        # test sha1, base64 = 1
        encode_base_sha224 = Encode("sha224", base64=1)
        sha224_base64_value = sha224_base64_value_list[0]
        self.assertEqual(encode_base_sha224.compute(value, pre_salt, postfit_salt), sha224_base64_value)

        # test sha224, base64 = 0
        encode_sha224 = Encode("sha224", base64=0)
        sha224_value = sha224_value_list[0]
        self.assertEqual(encode_sha224.compute(value, pre_salt, postfit_salt), sha224_value)

        # test sha512, base64 = 1
        encode_base_sha512 = Encode("sha512", base64=1)
        sha512_base64_value = sha512_base64_value_list[0]
        self.assertEqual(encode_base_sha512.compute(value, pre_salt, postfit_salt), sha512_base64_value)

        # test sha224, base64 = 0
        encode_sha512 = Encode("sha512", base64=0)
        sha512_value = sha512_value_list[0]
        self.assertEqual(encode_sha512.compute(value, pre_salt, postfit_salt), sha512_value)

        # test sha384, base64 = 1
        encode_base_sha384 = Encode("sha384", base64=1)
        sha384_base64_value = sha384_base64_value_list[0]
        self.assertEqual(encode_base_sha384.compute(value, pre_salt, postfit_salt), sha384_base64_value)

        # test sha384, base64 = 0
        encode_sha384 = Encode("sha384", base64=0)
        sha384_value = sha384_value_list[0]
        self.assertEqual(encode_sha384.compute(value, pre_salt, postfit_salt), sha384_value)

        # test is_support
        support_encode_method = ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]
        for method in support_encode_method:
            self.assertTrue(Encode.is_support(method))

        unsupport_method = "sha2"
        self.assertFalse(Encode.is_support(unsupport_method))

        # test conpute unsupport method
        test_compute = Encode("sha3840000", base64=0)
        self.assertEqual(test_compute.compute(value, pre_salt, postfit_salt), value)


if __name__ == '__main__':
    unittest.main()
