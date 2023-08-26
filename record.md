
编译 Paddle 的 cmake 指令，后两个是开启联合编译 CINN 的

```shell
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_TESTING=ON -DCINN_ONLY=OFF -DWITH_CINN=ON
```

编译参数指南：
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyixuanxiangbiao

