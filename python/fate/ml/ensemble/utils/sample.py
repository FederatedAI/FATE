from fate.arch.dataframe import DataFrame

def goss_sample(gh: DataFrame, top_rate: float, other_rate: float, random_seed=42):

    # check param, top rate + other rate <= 1, and they must be float
    assert isinstance(top_rate, float), "top rate must be float, but got {}".format(type(top_rate))
    assert isinstance(other_rate, float), "other rate must be float, but got {}".format(type(other_rate))
    assert top_rate + other_rate <= 1, "top rate + other rate must <= 1, but got {}".format(top_rate + other_rate)
    sample_num = len(gh)
    a_part_num = int(sample_num * top_rate)
    b_part_num = int(sample_num * other_rate)
    top_samples = gh.nlargest(n=a_part_num, columns=['g'], error=0)
    rest_samples = gh.drop(top_samples).sample(n=b_part_num, random_state=random_seed)
    sampled_rs = DataFrame.vstack([top_samples, rest_samples])
    return sampled_rs