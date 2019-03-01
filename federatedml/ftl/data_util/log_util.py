def create_shape_msg(components):
    msg = ""
    for c in components:
        msg += str(c.shape) + " "
    return msg
