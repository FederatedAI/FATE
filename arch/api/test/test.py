# s='apple    peach,banana    pear'
# li2=s.partition('    ')
# print(li2[0])
# print(li2[2])

# import pickle
# import json
# a = "dfasdfa"
# b = pickle.dumps(a)
# print(b)
# c = json.dumps(b)
# print(c)

def test(b=None):
    b = b or 11
    print(b)


test(b=22)
