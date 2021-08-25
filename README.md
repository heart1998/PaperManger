# PaperManger
This folder contains some papers on deep learning compilers, which are summarized as follows, and students in need can download and read them.
##1.安装组件说明 (以本人安装为例，适配版本) 
ubuntu 18.04 
LLVM version 10.0.0
TVM 0.7dev  (最新0.8版本会在cmake ..步骤有错误，0.7版本比较成熟，建议安装0.7dev）
cmake version 3.10.2 
python 3.8
conda 4.10.3
以及pycharm安装包，LLVM安装包，本人会附在文后
##2.TVM源码
因为tvm版本变化较大，v5.0-v6.0目录结构都不一样，所以安装要参照官方文档

[TVM官方文档](https://tvm.apache.org/docs/install/from_source.html)
2.1 从GitHub获取tvm源码下载
>git clone --recursive https://github.com/apache/tvm tvm

**注意加上recursive**

因为网络关系，我一直下不到tvm官网的安装包，所以我采用了手动下载的方式，具体来说，就是进入到github(www.github.com),然后搜索TVM的开源项目，找到第一个就是啦。因为我们要下载0.7dev,所以在tags找到0.7的版本进行下载.zip格式的文件。如图1:
![图1 0.7 版本tvm图示下载](https://upload-images.jianshu.io/upload_images/24938354-1d97550e542ab97a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
##2.构建共享库
**2.1 更新一下源,必须更新，不然安装依赖时会出错，打开终端，输入**
>sudo apt-get update

安装必要的依赖，这一步已经对cmake进行了安装。
>sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

**2.2 建立build选项**
进行tvm目录文件夹,首先创建一个build目录，复制 cmake/config.cmake到目录 **
>cd tvm
mkdir build
cp cmake/config.cmake build

此时，我们可以看到，在tvm文件夹下面出现了build的文件夹，中存在config.cmake的文件。
##3. LLVM下载
由于 LLVM 从源代码构建需要很长时间，您可以从以下位置下载 LLVM 的预构建版本 [LLVM 下载页面](http://releases.llvm.org/download.html) 。
![图2 llvm下载示意图](https://upload-images.jianshu.io/upload_images/24938354-0416918b121cb907.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####注解：因为LLVM下载也是比较费事的，所以我直接把下载好的LLVM解压后的文件夹和TVM0.7的源文件放在阿里云盘中，只需要下载到本地，放到需要安装的目录，需要的自取。有兴趣自己一步一步安装的，参考下面安装方法[ubuntu 安装LLVM](https://blog.csdn.net/qq_32460819/article/details/108449344)
注意配置llvm的全局环境，配置方法如下，终端输入：
>sudo vim ~/.bashrc
export PATH=$PATH:/home/xu/llvm/bin(输入自己llvm的bin路径)
source ~/.bashrc  (让环境生效)
llvm-config --version

xu@xu:~$ llvm-config --version
10.0.0
这样说明LLVM配置好了。


##4. 自定义编译选项
编辑 build/config.cmake自定义编译选项 ，打开config.cmake文件

1. （GPU配置，可选）如果，您想使用（OpenCL、RCOM、METAL、VULKAN 等）构建。找到 `set(USE_CUDA OFF)`改为`(USE_CUDA ON)`，即为启用 CUDA 后端。 对其他后端和库执行相同操作。
2. （方便debug)为了帮助调试，请确保已启用嵌入式图形执行器和调试功能 `set(USE_GRAPH_EXECUTOR ON)`和 `set(USE_PROFILER ON)`

3.   (配置LLVM，**必选**)TVM 需要 LLVM 用于 CPU 代码生成。 强烈建议您使用 LLVM 支持进行构建。


*   解压到某个位置，修改 `build/config.cmake`添加 `set(USE_LLVM /path/to/your/llvm/bin/llvm-config)`（自己llvm的存放绝对路径，也是LLVM编译通不过，路径查找不到问题解决方法）

*   也可以直接设置 `set(USE_LLVM ON)`并让 cmake 搜索可用版本的 LLVM
##5.编译
我们进入到tvm文件夹下
>cd build
cmake ..
make -j4（线程数，也可以设置8）

**注释：这一步会是大多数人出问题的地方，我当时困了好久在这，百度好多教程也找不到原因，最后在对编译问题log检查时，发现TVM本身下载的文件，会出现文件夹缺失的情况，我进入github找到缺失文件的目录，下载了缺失的文件夹，再次重新编译，完美通过！！！
[缺失文件及LLVM安装包等](https://www.aliyundrive.com/s/yNE3zqGJT7H)
 *    文件目录 TVM0.7dev     tvm/3rdparty 缺少4个文件夹
  ![图3 问题小结](https://upload-images.jianshu.io/upload_images/24938354-813fe6844a493a6d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


出现这个，说明我们编译成功，亲测0.8dev的tvm会卡在cmake ..这一步，所以我选择0.7dev的原因便在这。


![图3  编译成功图示](https://upload-images.jianshu.io/upload_images/24938354-1965b0cefe590ffd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
##6. Python包安装
tvm提供了两种方法，个人推荐第一种配置系统环境，比较简单，亲测可行。
#####方法一、此方法对于开发人员建议使用可能更改代码的  。

设置环境变量 PYTHONPATH 告诉 python 去哪里找 。 例如，假设我们克隆了 tvm 在目录中 /path/to/tvm 然后我们可以在添加以下行 ~/.bashrc 中 。 拉取代码并重建项目后，更改将立即反映出来。
```
export TVM_HOME=/path/to/tvm(自己tvm的路径)                    
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
```
#####方法二、（未测试）通过 安装 TVM python 绑定 setup.py ： 
```
# install tvm package for the current user
# NOTE: if you installed python via homebrew, --user is not needed during installaiton
#       it will be automatically installed to your user directory.
#       providing --user flag may trigger error during installation in such case.
export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
cd python; python setup.py install --user; cd ..

```
##7.安装Python依赖

请注意， --user如果您要安装到托管的本地环境，则不需要标志， 像 virtualenv.

必要的依赖（必选）：

    pip3 install --user numpy decorator attrs

 如果你想使用 RPC Tracker（可选）

    pip3 install --user tornado

如果要使用自动调整模块（可选，建议选择）

    pip3 install --user tornado psutil xgboost cloudpickle
到此，tvm安装部分结束，我们也可以选择安装anaconda（清华源）[ubuntu 18.04安装anaconda3](https://blog.csdn.net/Lucky_yw/article/details/89387073)和pycharm编辑器(可以在ubuntu软件商店进行下载)，相关的包我已经放在文中的链接中，有安装需求的可以自取。
##8.测试
在pycharm或者终端输入import tvm，然后打印版本号

```
import tvm
print(tvm.--version)


输出

0.7.0
```
至此，TVM安装完成，由于我没有安装CUDA，所以安装过程比较简单，如果你有TVM安装方面的任何问题，欢迎致信。下面将不定期更新有关TVM的相关内容，有兴趣的小伙伴也可以一起交流学习。
