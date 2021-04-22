# MMDNN_simple_example

Language : CN | [EN](./README.en.md) 

本文主要介绍了如何用<a herf='https://github.com/microsoft/MMdnn'>MMDNN</a>把Tensorflow模型转换为PyTorch模型

原始框架: TensorFlow

目标框架: PyTorch

快速示例：（首先下载*inception_v3.ckpt*并放到models文件夹下面）

也可以参考官方DEMO：<a herf='https://github.com/Microsoft/MMdnn/blob/master/docs/tf2pytorch.md'>https://github.com/Microsoft/MMdnn/blob/master/docs/tf2pytorch.md</a>

```sh
$python tf_save_model.py --checkpoint_path ./models/inception_v3.ckpt --output_path output_model/inception_v3.ckpt # generate .meta and .data
$cd output_model
$mmvismeta inception_v3.ckpt.meta ./logs/  # generate tesorboard file
$mmconvert -sf tensorflow -in inception_v3.ckpt.meta -iw inception_v3.ckpt --dstNode MMdnn_Output -df pytorch -om converted_pytorch.pth  # one-step converted
$cd ..
$python test.py  # test converted PyTorch model
# converted_pytorch.py and converted_pytorch.pth are the files we need 
```



## 1. 前提准备

### 1.0 安装

   ```bash
   $pip install mmdnn
   # or 
   $pip install -U git+https://github.com/Microsoft/MMdnn.git@master
   ```

   转换需要用到下列文件，有两种方式获取，一是直接用官方命令下载，二是自己生成。

   --model.ckpt.meta（网络结构）

   --model.ckpt.data（权重数据）

   ![image-20210421220158771](https://github.com/ylhz/MMDNN_simple_example/blob/main/readme_img/image-20210421220158771.png)

   如果有图中第一个和第三个文件，可以直接进入转换过程

   .ckpt文件是旧版本的权重文件，相当于.ckpt.data文件; .ckpt.meta可以用tensorboard显示其网络结构。

   注：inception_v3.ckpt和inception_v3.ckpt.data-00000-of-00001都用inception_v3.ckpt来读取

### 1.1 使用官方权重，下载后会直接生成相应文件

   ```bash
   $mmdownload -f tensorflow -n inception_v3
   ```
### 1.2 使用网络结构代码 + 权重文件生成

   ```python
   flow = tf.identity(logits_v3, name="MMdnn_Output")  
   ```

   需要定义最后输出层的名称，这里设置为了mmdnn官方默认的名字*'MMdnn_Output'*

   然后使用```saver.save(sess, model_path)```保存权重。

   保存文件如下：

   ![image-20210422090831223](https://github.com/ylhz/MMDNN_simple_example/blob/main/readme_img/image-20210422090831223.png)

## 2.转换

### 2.1 一步方法

   ```bash
   $ mmconvert -sf tensorflow -in inception_v3.ckpt.meta -iw inception_v3.ckpt --dstNode MMdnn_Output -df pytorch -om tf_to_pytorch_inception_v3.pth
   ```

   输出：tf_to_pytorch_inception_v3.pth

   参数说明：

   | 命令      | 含义             |
   | --------- | ---------------- |
   | -sf       | 输入模型类别     |
   | -in       | 输入模型网络结构 |
   | -iw       | 输入模型权重     |
   | --dstNode | 模型的输出节点   |
   | -df       | 输出模型类别     |
   | -om       | 输出模型名称     |

   

### 2.2 多步方法

   * TF模型转IR

   ```bash
   $mmtoir -f tensorflow -n inception_v3.ckpt.meta -w inception_v3.ckpt --dstNode outputs -o converted
   ```

   输出：*converted.json*，*converted.pb*，*converted.npy*
   

   * IR文件转PyTorch

   ```bash
   $mmtocode -f pytorch -n converted.pb -w converted.npy -d converted_pytorch.py -dw converted_pytorch.npy
   ```

   输出：*converted_pytorch.py*（转换后的pytorch版的网络结构代码），*converted_pytorch.npy*（转换后的网络权重）

   注：这就可以直接用这两个加载PyTorch版本的模型

   * 导出最终模型

   ```bash
   $mmtomodel -f pytorch -in converted_pytorch.py -iw converted_pytorch.npy -o converted_pytorch.pth
   ```

     输出：*converted_pytorch.pth*

## 3.使用

### 3.1 使用.pth文件

   需要用到**imp**库

   ```python
   import imp
   inc_v3 = torch.load('./converted_pytorch.pth')
   inc_v3 = inc_v3.cuda().eval()
   ```

### 3.2 使用.npy文件

   ```python
   inc_v3 = KitModel('./converted_pytorch.npy')
   ```
   
   (具体例子查看**test.py**)

## 遇到过的BUG:

* **MMdnn_Output is not in graph:**

  模型输出层没有命名为’MMdnn_Output‘，可以通过tensorboard检查网络结构。例子：*view_graph.py*

  也可通过以下命令查看：

  ```bash
  $mmvismeta inception_v3.ckpt.meta ./logs/  # generate tesorboard file
  $tensorboard --logdir=logs --port=6006 # open tensorboard
  ```

![image-20210422091652201](https://github.com/ylhz/MMDNN_simple_example/blob/main/readme_img/image-20210422091652201.png)


* 转换后的PyTorch和Tensorflow代码准确率不同

  可能是测试时没有设置model.eval()，或者是数据预处理方式不一致，比如没有把输入转化为[-1, 1]之间。（tensorflow一般是把输入转化为[-1, 1]之间）
  
  
