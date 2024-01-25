设备管理与内存分配
============

本章节介绍TORCH-TPU的设备注册与内存管理。

TORCH-TPU的设备注册
--------------------

在C++侧通过Pybinding将C++的相关函数封装成python可以调用的torch_tpu._C库文件。

在python侧通过“torch_tpu.__init__.py 注册相关的功能函数。

设备的管理通过继承“at::Allocator”类来实现，该类就是Pytorch当中进行设备的管理和内存的管理机制。

通过如下宏，可将实现tpu_alloc的绑定到PRIVATEUSEONE的后端上，实现设备的注册。

.. code-block:: shell

    REGISTER_ALLOCATOR ( PRIVATEUSEONE , &tpu_alloc )


Allocator类，其中主要包括以下几个函数:

.. code-block:: c++

    at::DataPtr allocate ( size_t nbytes );
    static void ReportAndDelete ( void * ptr );

在Pytorch当中，所有的运算都是在分配好的内存的基础上进行的，Pytorch的内存分配通过如下函数来实现:

.. code-block:: c++

    Tensor empty_strided_tpu ( IntArrayRef                size,
                           IntArrayRef                stride,
                           c10::optional<ScalarType>  dtype_opt,
                           c10::optional<Layout>      layout_opt,
                           c10::optional<Device>      device_opt,
                           c10::optional<bool>        pin_memory_opt )

该函数最终会调用到上述的tpu_alloc类。
