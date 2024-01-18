TORCH-TPU总体架构设计介绍
========

本章介绍 TORCH-TPU的总体设计。


TORCH-TPU总体架构设计
--------------------

TORCH-TPU前端设计


运行模式
~~~~~~~~~~~~~~~

TORCH-TPU根据运行时状态区分，TROCH-TPU可以运行于EagerMode和CompileMode两种模式下。
EagerMode即Pytorch的动态图执行模式。在该种模式下，每一次API的调用都会立即执行相应的计算。
CompileMode即Pytorch在2.0版本之后引入的Dynamo执行模式。在该模式下，会先建立
两种模式的使用方法，与Pytorch官方使用方法一致，详细不同可以参考后续文档说明。


EagerMode模式
--------------------

EagerMode模式具有灵活性高的优势，可以快速的进行实验和功能验证，下面是EagerMode模式的使用实例。

使用实例
~~~~~~~~~~~~~~~

基本原理
~~~~~~~~~~~~~~~

TODO原理图

CompileMode模式
--------------------

TODO