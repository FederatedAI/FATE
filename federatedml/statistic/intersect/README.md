### Introduction
#### RSA Intersection
This  folder contains code for implementing algorithm based on [RSA Intersection](https://books.google.com.hk/books?id=zfvf37_YS8cC&pg=PA73&lpg=PA73&dq=rsa+commutative+encryption&source=bl&ots=LbOiyIlr3E&sig=IIWlTGeoU0C8dRiN10uH2OAwobQ&hl=zh-CN&sa=X&ved=0ahUKEwiLoozC1tbXAhVDnJQKHbP7DvAQ6AEIdTAJ#v=onepage&q&f=false). This work is built on FATE, eggroll and federation API that construct the secure, distributed and parallel infrastructure.

Our Intersect module is trying to solve the problem that Privacy-Preserving Entity Match.
This module will help two parties to find the same user ids without leaking all their user ids 
to the other. This is illustrated in Figure 1. 

<div style="text-align:center", align=center>
<img src="./images/sample.png" alt="sample" width="500" height="250" /><br/>
Figure 1 RSA Intersection between party A and party B</div>

In Figure 1,Party A has user id u1,u2,u3,u5, while Party B has u1,u2,u3,u4. After Intersection,
party A and party B know their same user ids, which are u1,u2,u3, but party A know nothing about
other user ids of party B, like u4, and party B know nothing about party A except u1,u2,u3 as well.
While party A and party b transmit their processed id information to the other party, like Y<sub>B</sub> and Z<sub>A</sub>, 
it will not leak any raw ids. Z<sub>A</sub> can be safely because of the privacy key of party A. 
Each Y<sub>B</sub> includes different random value which binds to each value in X<sub>B</sub> and will be safely as well.

Using this module, we can get the intersection ids between two parties in security and efficiently.  

#### RAW Intersection
This intersection module implements the sample intersection method that A sends all his ids to B, and B will find the sample ids according to B's ids.Then B will send the intersection ids to A .

### Quick Start
You can refer to *example/intersect/README.md* to quickly start running intersection in standalone mode and cluster mode. 
