TPUStream是用于TPU设备的Stream机制，是c10::Stream的一个wrap。
通过TPUStream可以实现 c10::Stream和 sgrtStream_t的转换。

query():       x
synchronize(): v
stream();      ?
unwarp():      v
pack3():       v
unpack3():     v


如何单测？
