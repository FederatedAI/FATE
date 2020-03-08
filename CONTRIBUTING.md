# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. 

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read [Code of Conduct](CODE_OF_CONDUCT.md).
- Follow the instruction of [develop guide](./doc/develop_guide.md)
- Ensure you have signed the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).
- Check if my changes are consistent with the [guidelines](https://github.com/FederatedAI/FATE/blob/master/CONTRIBUTING.md#contributing-to-fate).
- Changes are consistent with the [PEP.8 Python Coding Style](https://www.python.org/dev/peps/pep-0008/).
- Run [unit tests and write examples](https://github.com/FederatedAI/FATE/blob/master/CONTRIBUTING.md#contribution-guidelines-and-standards).


## Contributing to FATE

### Developer Certificate of Origin

We are more than glad to accept your patches. To start with, please make sure you have signed every commit with Developer Certificate of Origin (DCO). If you are still not familiar with DCO, please check [this website](https://www.dita-ot.org/dco) for more information.

## Contributing code

1. If you want to purpose a new feature and implement it, please open an issue and discuss the design and implementation with us.

2. If you are interested in implementing an existed feature or bug-fix issue, please make a comment on the task that you want to work. And it is still strong recommended to discuss with us about your design and implementation.

3. If you are not sure where to start, trying smaller and easier issue may be a good idea and then take a look at the issue with the "contributions welcome" label.

Once you are ready to send your pull request, we will create a contributor branch to which you can send pull request. Then, FATE team members will be assigned to review and test your pull request. Once the review is pass, your contribution is accepted and will be released in a future version.

If you have ideas of new features but don't know how to start with, a [develop guide](./doc/develop_guide.md)  has been provided

## Contributing

### Contribution guidelines and standards

Before sending your pull request, here are some principle and standards to follow.

* Include unit tests when you contribute new features. You may created a test folder in your module and put your test files in it. Here is [an example](./federatedml/model_selection/test/).  FATE provides a [test script](./federatedml/test/run_test.sh) for you to check all unit test. Make sure your code work correctly.

* Provide examples in [example folder](./examples). If specific format data is needed, please also provided example data in [data folder](./example/data). In your algorithm examples, some dsl and conf files should be provided and provide a testsuite file with which user can test your example easily. Here is [an example of testsuite file](./examples/federatedml-1.x-examples/hetero_logistic_regression/hetero_lr_testsuite.json)

* After you contribute a new feature to FATE, FATE team members are (by default) responsible for the maintenance of this feature. This means we must make comparison between the benefit of contribution and the cost of maintaining the feature and accept those features with greater benefits.

### License
Include a license at the top of new files. A license example has list below.

```
#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
```

### Code Style

You should make sure your code and comments are consistent with the following coding style.

* [PEP.8 Python Coding Style](https://www.python.org/dev/peps/pep-0008/).

* [Numpy Docstring Format](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.
