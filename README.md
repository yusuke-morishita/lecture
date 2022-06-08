# Lecture

## 目次

- Python仮想環境の準備
- 学習データの準備


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


## 学習データの準備

### 顔画像の準備

最も有名な顔画像データベースの一つ "Labeled Faces in the Wild" (LFW) を使用する。

- 以下のURLから "All images as gzipped tar file" と書かれている箇所のファイル `lfw.tgz` をダウンロード
  - http://vis-www.cs.umass.edu/lfw/

- Pythonでの `lfw.tgz` の展開
  - 仮想環境の Python インタプリタで、以下を実行する。
```python
import tarfile
file = tarfile.open('lfw.tgz', 'r:gz')
file.extractall()
file.close
```

- Pythonでの顔画像の確認
  - OpenCV を使い、画像ファイルの読み込み、画像の表示を行う。
  - 仮想環境の Python インタプリタで、以下を実行する。
```python
import cv2
img = cv2.imread('lfw\Aaron_Guiel\Aaron_Guiel_0001.jpg')
cv2.imshow('image', img)
cv2.waitKey(-1)
```
