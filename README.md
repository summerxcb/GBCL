# GBCL

代码将于论文接受后发布


# 1. 更新系统
sudo apt update

# 2. 装 git
sudo apt install git -y

# 3. 装 Python 和 pip
sudo apt install python3 python3-pip -y

# 4. 装项目依赖
pip3 install numpy scikit-learn scipy psutil --break-system-packages

# 5. 拉代码
cd ~
git clone https://github.com/summerxcb/GBCL.git
cd GBCL

# 6. 打包 + 运行
python3 radar_detect_13.pyz
