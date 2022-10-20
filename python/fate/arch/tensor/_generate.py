if __name__ == "__main__":
    import pathlib

    _unary_path = pathlib.Path(__file__).parent.joinpath("_unary_ops.py")
    _binary_path = pathlib.Path(__file__).parent.joinpath("_binary_ops.py")
    _unary_funcs = [
        ("abs", "arc cosine"),
        ("asin", "arc sin"),
        ("atan", ""),
        ("atan2", ""),
        ("ceil", "ceiling"),
        ("cos", ""),
        ("cosh", ""),
        ("erf", "Gaussian error functiom"),
        ("erfinv", "Gaussian error functiom"),
        ("exp", ""),
        ("expm1", "exponential of each element minus 1"),
        ("floor", ""),
        ("frac", "fraction part 3.4 -> 0.4"),
        ("log", "natural log"),
        ("log1p", "y = log(1 + x)"),
        ("neg", ""),
        ("reciprocal", "1/x"),
        ("sigmoid", "sigmode(x)"),
        ("sign", ""),
        ("sin", ""),
        ("sinh", ""),
        ("sqrt", ""),
        ("square", ""),
        ("tan", ""),
        ("tanh", ""),
        ("trunc", "truncated integer"),
        ("rsqrt", "the reciprocal of the square-root"),
        ("round", ""),
    ]
    _binary_funcs = [
        ("add", ""),
        ("sub", ""),
        ("mul", ""),
        ("div", ""),
        ("pow", ""),
        ("remainder", ""),
        ("fmod", "element wise remainder of division"),
    ]
    with open(_unary_path, "w") as fw:
        fw.write("from ._ops import auto_unary_op\n")
        for name, comment in _unary_funcs:
            fw.write("\n")
            fw.write("\n")
            fw.write("@auto_unary_op\n")
            fw.write(f"def {name}(x, *args, **kwargs):\n")
            fw.write(f'    "{comment}"\n')
            fw.write(f"    ...\n")

    with open(_binary_path, "w") as fw:
        fw.write("from ._ops import auto_binary_op\n")
        for name, comment in _binary_funcs:
            fw.write("\n")
            fw.write("\n")
            fw.write("@auto_binary_op\n")
            fw.write(f"def {name}(x, y, *args, **kwargs):\n")
            fw.write(f'    "{comment}"\n')
            fw.write(f"    ...\n")
