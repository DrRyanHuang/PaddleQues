# PaddleQuestion


#### 1.新 IR 下的 Pass 是啥, 文档在哪里?

一个独立应用于图变换的模块机制，比如算子融合

#### 2.block 等概念是指？文档在哪里？


#### 3.OpDesc 是 Op 的 description ？还是说是一个算子对象? OpTranslator 返回的函数指针，将 OpDesc 翻译为 Operation.

```c++
using OpDesc = paddle::framework::OpDesc;

OpDesc(const std::string &type,
       const VariableNameMap &inputs,
       const VariableNameMap &outputs,
       const AttributeMap &attrs);
```

#### 4.`ir::IrContext` Context 是什么, 之前好像每个算子都有这个 Context 参数

#### 5.`pd.`命名空间是 paddle ?

`pd.`命名空间下的算子，对应于 PaddleDialect 下的算子体系，是通过yaml自动代码生成而来的，具体定义可以参见如下两个文件：

```
- build/paddle/fluid/ir/dialect/pd_op.h
- build/paddle/fluid/ir/dialect/pd_op.cc
```

#### 6.明确一下我的任务，mul 已经映射到了 matmul_with_flatten，`pd.matmul_with_flatten` 不存在，我是需要用 matmul 算子替换实现, 还是依旧用 matmul_with_flatten


(OpInfo `pd.matmul_with_flatten` 为啥不存在, `op_compat.yaml`不是有吗? 为啥 `ctx->GetRegisteredOpInfo` 中获取不到 `pd.matmul_with_flatten` )

```c++
const auto& op_info = ctx->GetRegisteredOpInfo(target_op_name);
```


```yaml
- op : matmul_with_flatten (mul)
  backward : matmul_with_flatten_grad (mul_grad)
  extra :
    attrs : [bool use_mkldnn = false, float scale_x = 1.0f, 'float[] scale_y = {1.0f}',
             float scale_out = 1.0f, bool force_fp32_output = false]
```

(为啥上边的 yaml 中，没有 inputs 和 outputs 这两项? 这个是和 `paddle/phi/kernels/cpu/matmul_kernel.cc` 下定义的一样吗)

我是需要看 `MatmulWithFlattenKernel` 的实现去找对应的属性吗?


报错内容是:
```
Op mul should have corresponding OpInfo pd.matmul_with_flatten
```

