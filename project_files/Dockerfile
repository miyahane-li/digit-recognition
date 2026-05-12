FROM python:3.10-slim

# 设置工作目录
WORKDIR /data/lxc

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制应用代码
COPY app.py .

# 运行应用
CMD ["python", "app.py"]