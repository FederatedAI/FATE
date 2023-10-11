from fate.arch.dataframe import DataFrame

def goss_sample(gh: DataFrame, top_rate: float, other_rate: float, random_seed=42):

    # check param, top rate + other rate <= 1, and they must be float
    assert isinstance(top_rate, float), "top rate must be float, but got {}".format(type(top_rate))
    assert isinstance(other_rate, float), "other rate must be float, but got {}".format(type(other_rate))
    assert top_rate + other_rate <= 1, "top rate + other rate must <= 1, but got {}".format(top_rate + other_rate)

    sample_num = len(gh)
    a_part_num = int(sample_num * top_rate)
    b_part_num = int(sample_num * other_rate)

    # sample top samples
    quantile_point = 1 - top_rate
    x = gh.quantile([quantile_point])['g'].iloc[0]
    print(x)
    df = gh['g']
    df_top_0 = df.iloc(df >= x)
    print('len 0', len(df_top_0))
    if len(df_top_0) == a_part_num:
        df_top = df_top_0
    elif len(df_top_0) > a_part_num:
        df_top_1 = df.iloc(df > x)
        print('len 1 {}'.format(len(df_top_1)))
        df_3 = df_top_0.drop(df_top_1).sample(n=a_part_num - len(df_top_1), random_state=random_seed)
        df_top = DataFrame.hstack([df_top_1, df_3])
        assert len(df_top) == a_part_num, "sample result len is {}, expected {}".format(len(df_top), a_part_num)
    else:
        raise RuntimeError('error sample result, >= {} got {} samples, while a part num is {}'
                           .format(quantile_point, len(df_top_0), a_part_num))
    # sample rest samples
    df_rest = gh.drop(df_top).sample(b_part_num, random_state=random_seed)

    sampled_rs = DataFrame.hstack([df_top, df_rest])
    return sampled_rs
