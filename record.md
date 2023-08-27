
编译 Paddle 的 cmake 指令，后两个是开启联合编译 CINN 的

```shell
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_TESTING=ON -DCINN_ONLY=OFF -DWITH_CINN=ON

make -j$(nproc)
```

编译参数指南：
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyixuanxiangbiao



摘自 Aurelius84

> 对于深度学习框架而言，在最初始阶段，通过枚举值对涉及到的类型进行简单的列举就可以满足需求。但是随着后续的发展，涉及到的类型种类越来越多，涉及到的优化也越来越细，对类型安全和类型推导的需求也越来越高，框架也开始向编译器方向靠拢。简单的枚举值类型定义已经逐步地无法深度学习框架的需求。




Type、Attrbute、Op

- Type 用来对类型进行抽象，比如 TensorType 、Fp32Type 等等。它不具有值的概念，比如 Fp32Type ，它表示一个变量的类型是 Float32 ，但是具体值是未知的。
- Attribute 是对一个具体的常量进行抽象，它是包含了具体的值的。比如 Fp32Attr ，用户是可以获取到它对应的数值是多少。
- Op 是对算子进行抽象。比如 ConvOp 、ReluOp 等等。它通过 yaml 文件进行定义，yaml 文件中会列举改 Op 的输入&输出&属性的约束，以及它所支持的 Trait 和 Interface 。IR会通过 python 脚本自动生成相应的 C++ 算子定义。
- 这三者是基于 Trait 和 Interface 进行定义的。它们会对关联自己所拥有的相应 Trait 和 Interface 。 比如 ConvOp 会关联 InferShapeInterface 、GradOpInterface、ReadOnlyTrait、ValueSemanticsTrait等特征和接口， Relu_Op则只会关联InferShapeInterface、GradOpInterface、InplaceTrait等特征和接口。
这三者也是可以任意扩展的，只要派生自相应的基类、关联相应的 Trait 和 Interface ， 遵循相应的实现规则即可。