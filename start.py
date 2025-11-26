import torch

x = torch.empty(5,3) # 5 * 3 的未初始化Tensor
print(x)

x = torch.zeros(12)#初始化零填充的矩阵
print(x)

x = torch.randn(5,3)#5*3随机填充矩阵
print(x)

x = torch.zeros(5, 3, dtype=torch.long) #5*3的long型0填充矩阵
print(x)

x = torch.tensor([5.5, 3])#直接根据数据创建
print(x)

#通过现有的tensor创建，默认宠用输入tensor的一些属性，例如数据类型
x = x.new_ones(5, 3, dtype = torch.float64)
print(x)

x = torch.randn_like(x, dtype = torch.float)# 指定新的数据类型
print(x)

print(x.shape) #获取tensor的形状
print(x.size())# 获取tensor的形状
print(x.numel())
#返回的torch.size是一个tuple,支持所有tuple的操作

#其他创建方法
# Tensor(*sizes)	基础构造函数
# tensor(data,)	类似np.array的构造函数
# ones(*sizes)	全1Tensor
# zeros(*sizes)	全0Tensor
# eye(*sizes)	对角线为1，其他为0
# arange(s,e,step)	从s到e，步长为step
# linspace(s,e,steps)	从s到e，均匀切分成steps份
# rand/randn(*sizes)	均匀/标准分布
# normal(mean,std)/uniform(from,to)	正态分布/均匀分布
# randperm(m)	随机排列

a = torch.arange(12).reshape(2, -1)
print(a)

#一些加法
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out = result)
print(result)

# 可以使用类似NumPy的索引操作来访问Tensor的一部分，需要注意的是：
# 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
print(x)
y = x[0, :]
print(y)
y += 1
print(y)
print(x[0, :])

#改变形状
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
# 注意view()返回的新Tensor与源Tensor虽然可能有不同的size，
# 但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。
# (顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)

# 如果我们想返回一个真正新的副本（即不共享data内存）该怎么办呢？
# Pytorch还提供了一个reshape()可以改变形状，
# 但是此函数并不能保证返回的是其拷贝，所以不推荐使用。
# 推荐先用clone创造一个副本然后再使用view
x_xp = x.clone().view(15)
x -= 1
print(x_xp)

#另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number：

x = torch.randn(1)
print(x)
print(x.item())

# 函数	功能
# trace	对角线元素之和(矩阵的迹)
# diag	对角线元素
# triu/tril	矩阵的上三角/下三角，可指定偏移量
# mm/bmm	矩阵乘法，batch的矩阵乘法
# addmm/addbmm/addmv/addr/baddbmm..	矩阵运算
# t	转置
# dot/cross	内积/外积
# inverse	求逆矩阵
# svd	奇异值分解

#python中的Tensor操作——官方文档https://pytorch.org/docs/stable/tensors.html

#broadcasting mechanism广播机制
#先适当复制元素使这两个Tensor形状相同后再按元素运算。例如：
x = torch.arange(1, 3).view(1, 2) #x是一行两列的矩阵
print(x)
y = torch.arange(1, 4).view(3, 1)#y是三行一列的矩阵
print(y)
print(x + y)

# 索引操作是不会开辟新内存的，而像y = x + y这样的运算是会新开内存的，
# 然后将y指向新内存。为了演示这一点，我们可以使用Python自带的id函数：
# 如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同。

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False

# 想指定结果到原来的y的内存，我们可以使用前面介绍的索引来进行替换操作。
# 在下面的例子中，我们把x + y的结果通过[:]写进y对应的内存中。
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True  索引操作不会开辟内存

# 我们还可以使用运算符全名函数中的out参数或者自加运算符+=(也即add_())达到上述效果，
# 例如torch.add(x, y, out=y)和y += x(y.add_(x))。
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True
# 虽然view返回的Tensor与源Tensor是共享data的，
# 但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），
# 二者id（内存地址）并不一致。

# 很容易用numpy()和from_numpy()将Tensor和NumPy中的数组相互转换。但是需要注意的一点是：
# 这两个函数所产生的的Tensor和NumPy中的数组共享相同的内存（所以他们之间的转换很快），
# 改变其中一个时另一个也会改变！！！2W