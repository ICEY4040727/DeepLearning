import numpy as np
import gzip #解包
import struct #加载解析数据
import os #添加路径到环境变量
import random #用来随机抽取图片验证

# 数据加载
class MNISTLoader:
    def __init__(self, data_dir='./mnist_data'):
        self.data_dir = data_dir
        self.train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        self.train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        self.test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        self.test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    def load_images(self, file_path):
        #加载并解析MNIST图像文件
        with gzip.open(file_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            #转换为Numpy数组
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, -1)
        return images.astype(np.float32) / 255.0  # 归一化到[0,1]

    def load_labels(self, file_path):
        #同上，加载并解析MNIST标签文件
        with gzip.open(file_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def get_data(self):
        #获取训练和测试数据
        print("Loading training data---")
        train_images = self.load_images(self.train_images_path)
        train_labels = self.load_labels(self.train_labels_path)
        print("Loading test data---")
        test_images = self.load_images(self.test_images_path)
        test_labels = self.load_labels(self.test_labels_path)
        return (train_images, train_labels), (test_images, test_labels)

# 神经网络
class Layer:
    def __init__(self, inputs, outputs, activation='ReLU'): 
        self.weights = np.random.randn(inputs, outputs) * 0.01  # Xavier初始化生成权重矩阵
        self.biases = np.zeros((1, outputs)) #偏置值
        self.activation = activation
        self.input = None #便于缓存
        self.output = None 

    def forward(self, input_data):
        self.input = input_data
        z = np.dot(input_data, self.weights) + self.biases #线性变换
        self.output = self._apply_activation(z) 
        return self.output 

    def _apply_activation(self, z):
        if self.activation == 'ReLU': #relu激活函数，适于避免数值过大过小，同时简化计算，适用于隐藏层
            return np.maximum(0, z)
        elif self.activation == 'Softmax':  #Softmax概率分布激活函数，适于最后一层输出
            exp_vals = np.exp(z - np.max(z, axis=1, keepdims=True))  #dimension维度形状不变
            return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            return z

class NeuralNetwork: 
    def __init__(self, layer_dims, activations):
        """初始化网络结构
        layer_dims: 维度
        activations: 各层激活函数
        """
        self.layers = []
        for i in range(len(layer_dims)-1):
            self.layers.append(Layer(layer_dims[i], layer_dims[i+1], activations[i]))

    def forward(self, X):
        #前向传播
        for layer in self.layers: #递归
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        #反向传播和参数更新
        m = X.shape[0]  #获取batch size用于平均化
        grads = []

        #输出层梯度
        output = self.layers[-1].output
        error = output - y
        d_weights = np.dot(self.layers[-2].output.T, error) / m    #权重梯度
        d_biases = np.sum(error, axis=0, keepdims=True) / m    #偏置梯度
        grads.insert(0, (d_weights, d_biases))

        #隐藏层梯度
        current_error = error    #初始当前误差
        for i in reversed(range(len(self.layers)-1)):  #遍历隐藏层
            #算激活函数的导数，用于链式法则
            activation_deriv = self._get_activation_deriv(
                self.layers[i].activation,
                self.layers[i].output
            )
            current_error = current_error.dot(self.layers[i+1].weights.T) * activation_deriv  #将误差反向传播到当前层，结合激活函数的导数。

            d_weights = np.dot(self.layers[i].input.T, current_error) / m  #当前层权重梯度
            d_biases = np.sum(current_error, axis=0, keepdims=True) / m
            grads.insert(0, (d_weights, d_biases))

        #参数更新
        for (dW, db), layer in zip(grads, self.layers):
            layer.weights -= learning_rate * dW
            layer.biases -= learning_rate * db   #梯度下降法更新当前梯度

    def _get_activation_deriv(self, activation, z):
        #获取激活函数的导数
        if activation == 'ReLU':
            return (z > 0).astype(float)
        elif activation == 'Softmax':
            return 1  # Softmax的导数在反向传播中特殊处理
        else:
            return np.ones_like(z)

#训练评估
def train(network, X_train, y_train, epochs=20, batch_size=128, lr=0.1):
   #训练网络并记录训练过程数据
    loss_history = []  
    accuracy_history = []  
    
    for epoch in range(epochs):
        #打乱数据
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0  #初始化当前epoch的损失累计值
        num_batches = X_train.shape[0] // batch_size
        
        #小批量训练
        for i in range(num_batches):
            X_batch = X_shuffled[i*batch_size:(i+1)*batch_size]
            y_batch = y_shuffled[i*batch_size:(i+1)*batch_size]          

            #前向传播
            output = network.forward(X_batch)           
            #计算损失（交叉熵）
            loss = -np.mean(np.sum(y_batch * np.log(output + 1e-15), axis=1))           
            #反向传播
            network.backward(X_batch, y_batch, lr)
            #累加损失
            epoch_loss += loss
        
        #计算平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        #计算准确率
        predictions = np.argmax(network.forward(X_train), axis=1)
        true_labels = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == true_labels)
        accuracy_history.append(accuracy)
        
        #打印训练进度
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    return loss_history, accuracy_history
# 主程序
if __name__ == "__main__":
    loader = MNISTLoader()
    (X_train, y_train), (X_test, y_test) = loader.get_data()
    
    # 转换为one-hot编码
    def one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels]
    
    y_train_onehot = one_hot(y_train)
    y_test_onehot = one_hot(y_test)

    # 网络初始化（输入784 → 隐藏层128 → 隐藏层64 → 输出10）
    network = NeuralNetwork(
        layer_dims=[784, 128, 64, 10],
        activations=['ReLU', 'ReLU', 'Softmax']
    )
    
    # 开始训练
    #train(network, X_train, y_train_onehot, epochs=20, batch_size=128, lr=0.1)
    
    #展示部分
import matplotlib.pyplot as plt

loss_history, accuracy_history = train(network, X_train, y_train_onehot, epochs=20, batch_size=128, lr=0.1)

# 绘制损失曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(accuracy_history, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 多次随机显示测试集图片及预测
while True:
    idx = random.randint(0, X_test.shape[0] - 1)
    img = X_test[idx]
    true_label = y_test[idx]
    pred_probs = network.forward(img.reshape(1, -1))
    pred_label = np.argmax(pred_probs)

    plt.figure()
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True label: {true_label} | Predicted: {pred_label}")
    plt.axis('off')
    plt.show()

    user_input = input("按回车显示下一张，输入其他任意内容退出：")
    if user_input.strip() != "":
        break