# 纵向皮尔逊相关系数

## 模块介绍

纵向皮尔逊相关系数模块用于计算特征列的皮尔逊相关系数。皮尔逊相关系数是两个变量$X$和$Y$的线性相关性的度量，定义如下：

$$\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X\sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y} = E\left[\left(\frac{X-\mu_X}{\sigma_X}\cdot\frac{Y-\mu_Y}{\sigma_Y}\right)\right]$$

令：

$$\tilde{X} = \frac{X-\mu_X}{\sigma_X}, \tilde{Y}=\frac{Y-\mu_Y}{\sigma_Y}$$

则有：

$$\rho_{X, Y} = E[\tilde{X}\tilde{Y}]$$

## 实现细节

我们使用了名为SPDZ的多方安全计算协议实现纵向皮尔逊相关系数模块。要了解更多细节，请参考： [[here](secureprotol.md)]

<!-- mkdocs
## Param

::: federatedml.param.pearson_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

## 如何使用

  部分参数

 
  - 列索引
      - 该参数取值为-1，或者一个int型数值列表。如取值为-1，所有变量列都将参与皮尔逊相关系数的计算；如取值为一个int型数值列表，则索引编号与列表中数值对应的列参与计算。
   
  - 列名称
      - 该参数取值为一个字符串型列表。列名出现在列表中的列将参与皮尔逊相关系数的计算。

  

!!! 提示

    如果同时设置了上述两个参数，则两参数限定参与计算的列的合集，将作为最终参与皮尔逊相关系数计算的列。

<!-- mkdocs
## Examples

{% include-examples "hetero_pearson" %}
-->
