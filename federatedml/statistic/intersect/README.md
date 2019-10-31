### Introduction
#### RSA Intersection
This folder contains code for implementing algorithm based on [RSA Intersection](https://books.google.com.hk/books?id=zfvf37_YS8cC&pg=PA73&lpg=PA73&dq=rsa+commutative+encryption&source=bl&ots=LbOiyIlr3E&sig=IIWlTGeoU0C8dRiN10uH2OAwobQ&hl=zh-CN&sa=X&ved=0ahUKEwiLoozC1tbXAhVDnJQKHbP7DvAQ6AEIdTAJ#v=onepage&q&f=false). This work is built on FATE, eggroll and federation API that construct the secure, distributed and parallel infrastructure.

Our Intersect module tries to solve the problem of Privacy-Preserving Entity Match.
This module helps two parties to find the same user ids without leaking user ids. This is illustrated in Figure 1.

<div style="text-align:center", align=center>
<img src="./images/sample.png" alt="sample" width="500" height="250" /><br/>
Figure 1 RSA Intersection between party A and party B</div>

In Figure 1,Party A has user id u1,u2,u3,u5, while Party B has u1,u2,u3,u4. After Intersection,
party A and party B know their overlapping user ids, which are u1,u2,u3, but party A knows nothing about
other user ids of party B, i.e. u4. Likewise, party B knows nothing about party A except that party A also has id u1,u2,u3.
While party A and party b transmit their processed id information to the other party, like Y<sub>B</sub> and Z<sub>A</sub>, 
it will not leak any raw ids. Z<sub>A</sub> can be safely because of the privacy key of party A. 
Each Y<sub>B</sub> includes different random value which binds to each value in X<sub>B</sub> and will be safely as well.

Using this module, we can get the intersection ids between two parties securely and efficiently.

#### RAW Intersection
This intersection module implements the sample intersection method that A sends all his ids to B, and B will find the sample ids according to B's ids.Then B will send the intersection ids to A .

#### Multi-Host Intersection
Both rsa and raw intersection support multi-host. It means a guest can do intersection with more than one host and finally get the common ID with all hosts. 
<div style="text-align:center", align=center>
<img src="./images/multi_hosts.png" alt="sample" width="500" height="250" /><br/>
Figure 2 multi-hosts Intersection</div>
As shown in Figure 2, guest first does intersection with each host and gets respective common IDs. Secondly, guest finds common IDs from all intersection results. Finally,
guest sends common IDs to every host if necessary.

### Quick Start
You can refer to *example/intersect/README.md* to quick start running intersection in standalone mode and cluster mode.

### Feature
Both RSA and RAW intersection supports the following features:
1. Support multi-host modeling task. The detail configuration for multi-host modeling task is located [Here](../../../doc/dsl_conf_setting_guide.md)

RSA intersection support the following extra features:
1. RSA support cache to speed up.

RAW intersection support the following extra features:
1. RAW support some encoders like md5 or sha256 to make it more safely.
