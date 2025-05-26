# LLM_comparation_report

## TASK1 ：完成 git clone 相关 git 的截图或部署完成的相关截图

### 见 src/git_screenshot 文件夹

## TASK2 ： 问答测试结果的相关截图

### 见 src/answer_screenshot 文件夹

## TASK3 : 大语言模型之间的横向对比分析

### 见 report.pdf 文件

## 补充 ：模型搭建教程

### STEP1 ： 环境搭建

#### 下载并激活 conda

```bash
wegt https:repo.anaconda.com/minconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
conda create -n qwen_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```

#### 安装依赖

```bash
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -U pip setuptools wheel
pip install "intel-extension-for-transformers==1.4.2" "neural-compressor==2.5" "transformers==4.33.3" "modelscope==1.9.5" "pydantic==1.10.13" "sentencepiece" "tiktoken" "einops" "transformers_stream_generator" "uvicorn" "fastapi" "yacs" "setuptools_scm"
pip install tqdm huggingface-hub
```

### STEP2 : 运行模型，获得回答

```bash
cd /mnt/data
git clone "模型http下载path"
cd /mnt/workspace
python run_qwen_cpu.py
```

#### 在运行 python 文件之前要在工作目录中 add new file;

#### 粘贴推理脚本代码，注意修改 model_name 为 "mnt/data/filename"
