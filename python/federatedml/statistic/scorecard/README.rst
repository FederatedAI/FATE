
Scorecard
=========

Introduction
------------

A `credit scorecard <https://en.wikipedia.org/wiki/Credit_scorecards>`_ is
a credit model for measuring individuals' creditworthiness.
By quantifying the probability that a lender may display a defined behavior,
scorecards represents the lender's creditworthiness in numeric credit score.

Scorecard module of FATE provides a score transformer which scales
predict score(probability of default) to credit score
with user-defined range and parameter values.

Param
------

.. automodule:: federatedml.param.scorecard_param
   :members:

How to Use
----------

:params:

    :method: score method, currently only supports "credit"

    :offset: score baseline, default 500

    :factor: scoring step, when odds double, result score increases by this factor, default 20

    :upper_limit_ratio: upper bound for odds ratio, credit score upper bound is upper_limit_ratio * offset, default 3

    :lower_limit_value: lower bound for result score, default 0

    :need_run: Indicate if this module needed to be run, default True

:examples:
    There is an example :download:`[conf] <../../../../examples/dsl/v2/scorecard/test_scorecard_job_conf.json>`
    and :download:`[dsl] <../../../../examples/dsl/v2/scorecard/test_scorecard_job_dsl.json>`
