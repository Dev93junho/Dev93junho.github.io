# basic Deep Learning
## Perceptron
- AND Gate, NAND Gate, OR Gate, XOR Gate
- Perceptron
- MLP::Muti Layer Perceptron
## Neural Net
### Activation Function
01. sigmoid
02. Leaky ReLU
03. tanh
04. Maxout
05. ReLU
06. ELU
07. Swish
08. SoftMax
## Back Propagate
## SGD
## Batch Normalize
## Overfitting, DropOut
## Hyper parameter
## Convolutional Neural Network
<pre>
<code>
from tensorflow.keras import Conv2D
Conv2D(filters, kernal_size, strides, padding, activation, input_shape)
- filters: layer에서 나갈 때 몇 개의 filter를 만들 것인지 (a.k.a weights, filters, channels)
- kernel_size: filter(Weight)의 사이즈
- strides: 몇 개의 pixel을 skip 하면서 훑어지나갈 것인지 (사이즈에도 영향을 줌)
- padding: zero padding을 만들 것인지. VALID는 Padding이 없고, SAME은 Padding이 있음 (사이즈에도 영향을 줌)
- activation: Activation Function parameter. 당장 설정 안해도 Layer층을 따로 만들 수 있음. relu, sigmoid 등 layer 특성에 따라 다른 function 사용
<pre>
<code>


## Let's Start Deep Learning
#### DNN
#### Image Net
#### VGG
#### GoogLeNet
#### ResNet

#### Reference

