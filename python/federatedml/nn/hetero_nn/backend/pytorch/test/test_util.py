def dump_obj(obj, path):
    import pickle
    pickle.dump(obj, open(path, 'bw'))
