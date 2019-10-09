import pickle
import codecs


def obj_to_pickle_string(x, file_path=None):
    if file_path is not None:
        print("save model to file")
        output = open(file_path, 'wb')
        pickle.dump(x, output)
        return file_path
    else:
        print("turn model to byte")
        x = codecs.encode(pickle.dumps(x), "base64").decode()
        print(len(x))
        return x
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    if ".pkl" in s:
        df = open(s, "rb")
        print("load model from file")
        return pickle.load(df)
    else:
        print("load model from byte")
        return pickle.loads(codecs.decode(s.encode(), "base64"))
