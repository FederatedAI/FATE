import six
import abc
import functools


@six.add_metaclass(abc.ABCMeta)
class Aggregator(object):

    @classmethod
    def init(cls):
        return None

    @classmethod
    def op(cls, a, b):
        pass

    @classmethod
    def agg(cls, values):
        if cls.init() is None:
            return cls.post_op(functools.reduce(cls.op, values))
        else:
            return cls.post_op(functools.reduce(cls.op, values, cls.init()))

    @classmethod
    def post_op(cls, x):
        return x


class MeanAggregator(Aggregator):

    @classmethod
    def init(cls):
        return dict(), 0.0

    @classmethod
    def op(cls, model_and_weight1, model_and_weight2):
        agg_model, agg_weight = model_and_weight1
        model, weight = model_and_weight2
        new_agg_weight = agg_weight + weight
        for k in model.keys():
            if k in agg_model:
                agg_model[k] = agg_model[k] * agg_weight / new_agg_weight + weight * model[k] / new_agg_weight
            else:
                agg_model[k] = model[k]
        return agg_model, new_agg_weight

    @classmethod
    def post_op(cls, x):
        return x[0]


def main():
    s = zip([{1: 2, 3: 4}, {1: 3, 3: 7}], [1, 2])
    agg = MeanAggregator.agg(s)
    print(agg)
    print(s)


if __name__ == '__main__':
    main()
