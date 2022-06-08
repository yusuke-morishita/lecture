# Lecture

## Python仮想環境の準備

Windows上のPython（公式版／Anaconda）で仮想環境を準備し、必要なパッケージをインストールする。

- Pythonのインストール
  - Python 公式サイトからダウンロードし、インストール。
    - https://www.python.org/downloads/windows/
    - 2022年6月1日時点の最新版は3.10.4

- フォルダの作成
```bat
mkdir lecture1
cd lecture1
```

- 仮想環境の作成
  - 仮想環境名は任意。ここでは`.venv`とする。
```bat
REM Python公式の場合
python -m venv .venv

REM Anacondaの場合
conda create -n .venv python
```

- 仮想環境の起動
  - 起動すると、コマンドプロンプトの先頭に仮想環境名が`(.venv)`のように表示される。
```bat
REM Python公式の場合
.venv\Scripts\activate

REM Anacondaの場合
conda activate .venv
```

- パッケージのインストール
  - 仮想環境内でPyTorch, Torchvision, OpenCVをインストールする。
```bat
REM Python公式の場合
pip install torch torchvision opencv-python

REM Anacondaの場合
conda install pytorch torchvision -c pytorch
conda install opencv
```
