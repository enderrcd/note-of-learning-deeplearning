# 《动手学深度学习》学习笔记

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-更新中-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/github/stars/enderrcd/note-of-learning-deeplearning?style=social" alt="stars">
  <img src="https://img.shields.io/github/last-commit/enderrcd/note-of-learning-deeplearning" alt="last commit">
</p>

<p align="center">
  <i>理解深度学习的最佳方法是学以致用。——《动手学深度学习》</i>
</p>

<p align="center">
  <a href="#-项目简介"> 项目简介</a> •
  <a href="#-笔记目录"> 笔记目录</a> •
  <a href="#-环境配置"> 环境配置</a> •
  <a href= "#-贡献指南">贡献指南</a> •
  <a href= "#-参考资料">参考资料</a> •
  <a href= "#-致谢">致谢</a> •
</p>

---

##  项目简介

这个仓库就是本人学习《动手学习深度学习》这本书的笔记，喜欢的话可以简单看看
> 🔗 **官方资源**：[《动手学深度学习》网站](https://zh.d2l.ai/) | [官方GitHub](https://github.com/d2l-ai/d2l-zh) | [讨论论坛](https://discuss.d2l.ai/c/chinese-version/16)

---

## 笔记目录

<details open>
<summary><b>点击展开/折叠完整目录</b></summary>

<br>

 - [线性回归](./线性回归.ipynb)
 - [softmax回归](./softmax回归.ipynb)
 - [多层感知机](./多层感知机.ipynb)
 - [深度学习计算](./深度学习计算.ipynb)
 - [卷积神经网络](./卷积神经网络.ipynb)
 - [现代卷积神经网络](./现代卷积神经网络.ipynb)

</details>

---

##  环境配置

### 使用 conda 创建环境

```bash
# 1. 创建Python 3.9虚拟环境（d2l 0.17.6推荐版本）
conda create -n d2l python=3.9 -y
conda activate d2l

# 2. 升级pip（避免安装问题）
pip install --upgrade pip
```
### 安装PyTorch,在这里要注意自己的显卡型号，下载对应的cuda.这里以RTX40系列为例子
```bash

# RTX 4060搭配CUDA 12.8，选择兼容的PyTorch 2.1.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 验证GPU可用性
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

```

### 理论上这时候应该环境就配好了，但是实际上pip check后会有一堆依赖缺失，例如
```bash
# d2l 0.17.6有严格的版本要求，必须按照以下顺序安装

# 3.1 安装基础科学计算库
pip install numpy==1.21.5
pip install requests==2.25.1
pip install six==1.16.0
pip install certifi==2022.12.7
pip install charset-normalizer==2.1.1
pip install urllib3==1.26.14
pip install chardet==4.0.0
pip install idna==2.10

# 3.2 安装PyTorch依赖
pip install mpmath==1.3.0
pip install sympy==1.10.1
pip install filelock==3.9.0
pip install networkx==2.8.8
pip install pillow==9.4.0
pip install jinja2==3.1.2
pip install markupsafe==2.1.2
pip install fsspec==2023.3.0

# 3.3 安装数据科学库
pip install python-dateutil==2.8.2
pip install pytz==2022.7
pip install packaging==23.0
pip install pandas==1.2.4

# 3.4 安装Matplotlib及其依赖
pip install contourpy==1.0.7
pip install cycler==0.11.0
pip install fonttools==4.39.0
pip install kiwisolver==1.4.4
pip install pyparsing==3.0.9
pip install matplotlib==3.5.1


# 4.1 Jupyter核心
pip install ipykernel==6.21.0
pip install ipython==8.10.0
pip install jupyter-client==7.4.9
pip install jupyter-core==5.3.0
pip install traitlets==5.9.0
pip install pyzmq==25.0.2
pip install tornado==6.2
pip install psutil==5.9.4

# 4.2 Jupyter组件
pip install ipywidgets==8.0.4
pip install jupyter-console==6.4.4
pip install nbconvert==7.2.9
pip install notebook==6.5.3
pip install qtconsole==5.4.2
pip install jupyter==1.0.0

# 4.3 Jupyter依赖补全
pip install comm==0.1.3
pip install debugpy==1.6.7
pip install matplotlib-inline==0.1.6
pip install jupyterlab-widgets==3.0.7
pip install widgetsnbextension==4.0.7
pip install prompt-toolkit==3.0.38
pip install pygments==2.14.0
pip install beautifulsoup4==4.11.2
pip install soupsieve==2.4
pip install bleach==6.0.0
pip install webencodings==0.5.1
pip install defusedxml==0.7.1
pip install importlib-metadata==6.0.0
pip install zipp==3.15.0
pip install jupyterlab-pygments==0.2.2
pip install mistune==2.0.5
pip install nbclient==0.7.3
pip install nbformat==5.7.3
pip install fastjsonschema==2.17
pip install jsonschema==4.17.3
pip install attrs==23.1.0
pip install pyrsistent==0.19.3
pip install pandocfilters==1.5.0
pip install tinycss2==1.2.1


# 5.1 Notebook服务器组件
pip install argon2-cffi==21.3.0
pip install argon2-cffi-bindings==21.2.0
pip install cffi==1.15.1
pip install pycparser==2.21
pip install ipython-genutils==0.2.0
pip install nbclassic==0.5.3
pip install jupyter-server==2.6.0
pip install notebook-shim==0.2.3
pip install anyio==3.7.1
pip install exceptiongroup==1.1.2
pip install sniffio==1.3.0
pip install jupyter-events==0.6.3
pip install python-json-logger==2.0.7
pip install pyyaml==6.0
pip install rfc3339-validator==0.1.4
pip install rfc3986-validator==0.1.1
pip install jupyter-server-terminals==0.4.4
pip install overrides==7.3.1
pip install websocket-client==1.5.2
pip install nest-asyncio==1.5.6
pip install prometheus-client==0.16.0
pip install send2trash==1.8.0
pip install terminado==0.17.1
pip install pywinpty==2.0.10

# 5.2 IPython增强
pip install backcall==0.2.0
pip install colorama==0.4.6
pip install decorator==5.1.1
pip install jedi==0.18.2
pip install parso==0.8.3
pip install pickleshare==0.7.5
pip install stack-data==0.6.2
pip install asttokens==2.2.1
pip install executing==1.2.0
pip install pure-eval==0.2.2
pip install wcwidth==0.2.6
pip install platformdirs==3.5.1
pip install pywin32==305
pip install entrypoints==0.4

```
不限于上述所说的问题
### 最后安装d2l包，并验证环境
```bash
# 6. 安装d2l
pip install d2l==0.17.6

# 7. 完整验证
python -c "
import torch
import d2l
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

print('✅ PyTorch版本:', torch.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
print('✅ GPU型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')
print('✅ d2l版本:', d2l.__version__)
print('✅ numpy版本:', np.__version__)
print('✅ pandas版本:', pd.__version__)
print('✅ requests版本:', requests.__version__)
"

# 8. 启动Jupyter
jupyter notebook

```


##  贡献指南

欢迎提交PR或Issue！如果你发现任何错误或有改进建议：

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

##  参考资料

- [《动手学深度学习》官方中文版 - 在线阅读](https://zh.d2l.ai/) 
- [官方GitHub仓库 - d2l-zh - 官方代码](https://github.com/d2l-ai/d2l-zh)
- [PyTorch官方文档 - PyTorch API参考](https://pytorch.org/docs/stable/index.html) 

### 推荐阅读
- [李沐的深度学习课程 - B站视频](https://space.bilibili.com/1567748478/channel/detail?cid=175509) 

##  致谢

- 感谢 [@MuLi](https://github.com/mli) 和 [@astonzhang](https://github.com/astonzhang) 等作者提供的优秀教材
- 感谢所有为《动手学深度学习》做出贡献的开发者
- 感谢PyTorch团队提供的优秀深度学习框架

---

<p align="center">
  <b>如果这个笔记对你有帮助，请给一个star吧！</b>
  <br>
  <br>
  <a href="https://github.com/enderrcd/note-of-deeplearning">
    <img src="https://img.shields.io/github/stars/enderrcd/note-of-deeplearning?style=for-the-badge&logo=github" alt="GitHub stars">
  </a>
</p>
