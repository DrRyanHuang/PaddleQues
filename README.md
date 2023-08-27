# PaddleQuestion


### 202308xx 11:00

#### 1. GraphCompiler 作用是啥?



#### 2. 对齐 无环 是什么意思?

> 1.3 推理方向
> 主要是希望模型结构能够跟业界对齐（无环）。在此基础上，让 Pass 更加的完备、简洁。
>
> 要做到这一点，一是对现有模型数据结构升级改造，从数据结构上保证无环。同时，对算子定义进行完善升级，保证类似 inplace 算子的合理语义表示。在做到这些的基础上，进一步对 Pass 进行升级，保证完备性、稳定性等等。


#### 3. Dialect 的概念是否可以明确一下，和 Context 类似？

IR 框架作为一个最底层的动态库，提供算子、类型等的基础设施和注册接口，由上层按需使用。
同时，IR 库提供内置 Dialect, 注册一些最常见的类型和算子。

> Dialect 用来对 Type、Attribtue、Op做模块化管理， 比如 BuiltinDialect、PaddleDialect、CinnDialect等等。一个Dialect 里面包含了一系列的 Type、Attribtue、Op的定义。相应的，每个 Type、Attribtue、Op都是定义在某个唯一的 Dialect 里面。对整个 IR 框架而言， Dialect 是可以随意插拔的。
>
> Dialect 也是可以任意扩展的。只要派生自相应的基类、定义所以所拥有的 Type 、 Attrbute 、Op 即可。
>
> 这一层是 IR 适应多种场景的基础。这一层的每一个要素都是可定制化扩展的，一般情况下，针对一个具体的场景，比如分布式、编译器。都需要定义自己需要用到的Trait、Interfce，然后定义自己的Dialect，在自己的Dialect里面，定义自己需要用到的 Type、Attribute、Op。



#### 4. (想问下这个宏就类似 switch 语句吗)

```c++
// 转换通过先判断typeid，以C++作为桥梁进行判断
template <typename T>
inline bool IsType(const std::type_index& type) {
  return type == typeid(T);
}

#define PD_FOR_EACH_DATA_TYPE(_)      \
  _(bool, DataType::BOOL)             \
  _(int8_t, DataType::INT8)           \
  _(uint8_t, DataType::UINT8)         \
  _(int16_t, DataType::INT16)         \
  _(uint16_t, DataType::UINT16)       \
  _(int32_t, DataType::INT32)         \
  _(uint32_t, DataType::UINT32)       \
  _(int64_t, DataType::INT64)         \
  _(uint64_t, DataType::UINT64)       \
  _(bfloat16, DataType::BFLOAT16)     \
  _(float16, DataType::FLOAT16)       \
  _(float, DataType::FLOAT32)         \
  _(double, DataType::FLOAT64)        \
  _(complex64, DataType::COMPLEX64)   \
  _(complex128, DataType::COMPLEX128) \
  _(pstring, DataType::PSTRING)

```

#### 5. lod_level 的含义

比如 DenseTensorType 需要传递 shape、lod_level、data_type作为参数


#### 6. isa 的意思是?

判断两个类型相等， 直接用相等运算符。这个主要用来做类型验证，比如 Op 定义中约束了算子的输入类型。可通过该接口判定类型是否满足约束。
```
Type. type1, type2;
........
if(type1 == type2) {
......}

if(type1 != type2) {
.......}
```

判断是否是某种类型, 用 isa 接口(关于 isa 接口和下文的 dyn_cast 接口的实现，在后文的具体实现中会讲到)
```
Type type1;
.....

// type的impl指针里面存储了AbstractType*指针，里面有TypeID对象，所以只需要判断该TypeID和 LoDTensorType的TypeID是否一致即可实现isa接口。
if(type1.isa<LoDTensorType>()) {
   .....
}
```

#### 7. 计算图有环怎么理解, UD链 是什么

一方面，**计算图** 有环。
另一方面，效率也不高，要想知道一个变量都被哪些算子关联了，就必须遍历 block 中所有算子的所有输入输出，进行字符串比对。
在 Graph 中，一方面，变量和算子被同时抽象为了计算图节点，这增加了图优化的复杂度。
另一方面，Graph 内嵌了 Program ，图优化不仅要处理图节点的UD链，还得处理图节点内嵌的 OpDesc & VarDesc 的 UD链，进一步增加了图优化的复杂度。编译器相对前沿一些，它的 Grpah 里面没有内嵌 Program 。 它也严格明确了，计算图中算子为节点，变量为边。


### 20230825 11:00

#### 1.新 IR 下的 Pass 是啥, 文档在哪里?

一个独立应用于图变换的模块机制，比如算子融合，在目录`paddle/ir/pass/`下

比如 `dropout p=0` 时, 就可以删除掉 dropout

感兴趣可以查看 Pass 类的 ApplyImpl(graph) 虚函数

#### 2.block region 等概念是指？文档在哪里？

CINN 旧IR 都有, (计算图表示 ProgramDesc) block 有很多 Op, 更像是一个管理容器，block 不嵌套，里面会有控制流

region 类似作用域

#### 3.OpDesc 是 Op 的 description ？还是说是一个算子对象? OpTranslator 返回的函数指针，将 OpDesc 翻译为 Operation.

```c++
using OpDesc = paddle::framework::OpDesc;

OpDesc(const std::string &type,
       const VariableNameMap &inputs,
       const VariableNameMap &outputs,
       const AttributeMap &attrs);
```

OpDesc 会和 OpProto 相对应，OpDesc 旧IR, Operation是 新IR

旧IR: OpMaker 和 Kernel , 前者是数据输入的描述, 是一个表述层, 后者计算层，怎么支持不同计算设备(XPU)，计算逻辑，硬件分发, 前者类似是前端，后者类似后端，一个 Op 会有多个 kernel

OpTranslator 是 旧IR 转 新IR


#### 4.`ir::IrContext` Context 是什么, 之前好像每个算子都有这个 Context 参数

是一个中央管理器，管理 Dialect type attr，所有的存储都会放到这里

#### 5.`pd.`命名空间是 paddle ?

`pd.`命名空间下的算子，对应于 PaddleDialect 下的算子体系，是通过yaml自动代码生成而来的，具体定义可以参见如下两个文件：

```
- build/paddle/fluid/ir/dialect/pd_op.h
- build/paddle/fluid/ir/dialect/pd_op.cc
```

#### 6.明确一下我的任务，mul 已经映射到了 matmul_with_flatten，`pd.matmul_with_flatten` 不存在，我是需要用 matmul 算子替换实现, 还是依旧用 matmul_with_flatten


(OpInfo `pd.matmul_with_flatten` 为啥不存在, `op_compat.yaml`不是有吗? 为啥 `ctx->GetRegisteredOpInfo` 中获取不到 `pd.matmul_with_flatten` )
( 可能 yaml 中 matmul_with_flatten (mul) 这种语法有些问题? )

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
可以查看 `fluid/operators/mul_op/cc` 来看对应的属性, 但是新的 IR 要求与 Kernel 对齐, 不过当前题目 operators 签名和 kernels 签名一致，

报错内容是:
```
Op mul should have corresponding OpInfo pd.matmul_with_flatten
```

