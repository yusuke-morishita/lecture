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
# OpenCVを使うことの宣言
import cv2
# 画像ファイルの読み込み
img = cv2.imread('lfw\Aaron_Guiel\Aaron_Guiel_0001.jpg')
# 読み込んだ画像の表示
cv2.imshow('image', img)
cv2.waitKey(-1)
```

### 学習データの準備：入力画像

OpenCVに実装されている顔検出を使い、LFWの顔画像から顔の領域を切り出す。

- OpenCVを使った顔検出について
  - OpenCV に実装されている「Haar Cascade 顔検出」を利用する。2001年提案の非常に高速で代表的な手法。
  - 顔検出用の学習モデルを、以下のURLからダウンロードし、保存する
    - https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    - 上記はGithubのページ。ファイルを（wget等で）直接ダウンロードする場合は、以下
      - https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

- PythonでのOpenCVを使った顔検出
  - OpenCV を使い、読み込んだ顔画像から顔検出を行い、結果を表示する。さらに、顔領域を切り出した画像を生成する。
  - 仮想環境の Python インタプリタで、以下を実行する。
```python
# OpenCVを使うことの宣言
import cv2
# 画像ファイルの読み込み
img = cv2.imread('lfw\Aaron_Guiel\Aaron_Guiel_0001.jpg')
# 画像をグレースケールに変換
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出を初期化（学習モデルをファイルから読み込み）
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 顔検出を実行
faces = detector.detectMultiScale(img_g)
# 顔検出の結果を表示
print(faces)
# [[66 67 119 119]]などのように顔検出の結果が表示される
# 顔検出結果には、顔の矩形の [左上x座標、左上y座標、矩形幅、矩形高さ] が入る

# 顔検出結果の取り出し
x, y, w, h = faces[0]
# 検出した顔の矩形を入力画像に描画
#  - OpenCVの矩形を描画する関数cv2.rectangle()を使う
#  - 矩形の左上座標(x,y)、右下座標(x+w, y+h)、赤色(0,0,255)、線幅2を指定して描画
cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

# 顔検出結果を描画した画像の表示
cv2.imshow('image', img)
cv2.waitKey(-1)

# 検出した顔の位置で顔領域の画像を切り出し
#  - OpenCVの画像を拡大・縮小する関数cv2.resize()を使う
#  - グレースケール画像（numpyの2次元配列）から、0次元目＝画像の縦方向を[y:y+h]の範囲、1次元目＝画像の横方向を[x:x+w]の範囲で切り出す：img_g[y:y+h, x:x+w]
#  - 切り出した範囲を、cv2.resize()関数を使い(32,32)画素の大きさに縮小し、学習データの画像サイズを揃える
img_n = cv2.resize(img_g[y:y+h, x:x+w], dsize=(32,32))

# 顔領域を切り出した画像の表示
cv2.imshow('image', img_n)
cv2.waitKey(-1)
```
