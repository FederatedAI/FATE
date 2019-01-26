### Introduction

This  folder contains code for implementing algorithm based on [RSA Intersection](https://books.google.com.hk/books?id=zfvf37_YS8cC&pg=PA73&lpg=PA73&dq=rsa+commutative+encryption&source=bl&ots=LbOiyIlr3E&sig=IIWlTGeoU0C8dRiN10uH2OAwobQ&hl=zh-CN&sa=X&ved=0ahUKEwiLoozC1tbXAhVDnJQKHbP7DvAQ6AEIdTAJ#v=onepage&q&f=false). This work is built on Fate, eggroll and federation API that construct the secure, distributed and parallel infrastructure.

Our Intersect module is trying to solve the problem that Privacy-Preserving Entity Match.
This module will help two parties to find the same user ids without leaking all their user ids 
to the other. This is illustrated in Figure 1. 

<div style="text-align:center", align=center>
<img src="./images/sample.png" alt="sample" width="500" height="250" /><br/>
Figure 1ï¼?RSA Intersection between party A and party B</div>

In Figure 1,Party A has user id u1,u2,u3,u5, while Party B has u1,u2,u3,u4. After Intersection,
party A and party B know their same user id, which are u1,u2,u3, but party A know nothing about
other user id of party B, like u4, and party B know nothing about party A except u1,u2,u3 as well.
 
> Each commutation between party A and party B has owned privacy key.(e.g. d in partyA and random r in party B)

### Quick Start
You can refer to *example/intersect/README.md* to quickly start running intersection in standalone mode and cluster mode. 