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
#  - グレースケール画像から、画像の縦方向を[y:y+h]の範囲、画像の横方向を[x:x+w]の範囲で切り出す
#  - 切り出した範囲を、cv2.resize()関数を使い(32,32)画素の大きさに縮小し、学習データの画像サイズを揃える
img_n = cv2.resize(img_g[y:y+h, x:x+w], dsize=(32,32))

# 顔領域を切り出した画像の表示
cv2.imshow('image', img_n)
cv2.waitKey(-1)
```

### 学習データの準備：正解データ

LFWの顔画像（一部）に対して、予め笑顔か無表情かのラベルを付与した正解データを使用する。

- 正解データのダウンロード
  - 以下のファイルをダウンロードし、保存する
    - https://github.com/yusuke-morishita/lecture/blob/main/1/lfw_with_smile_label.txt
    - 上記はGithubのページ。ファイルを（wget等で）直接ダウンロードする場合は、以下
      - https://raw.githubusercontent.com/yusuke-morishita/lecture/main/1/lfw_with_smile_label.txt

- 正解データのファイルの内容
  - 画像ファイル名、笑顔ラベル（笑顔=1、無表情=0）の順で、1行ずつ格納
  - 無表情が 1,913 枚、笑顔が 1,867 枚、合計 3,780 枚
```
.\lfw\AJ_Lamas\AJ_Lamas_0001.jpg 1
.\lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg 0
.\lfw\Aaron_Peirsol\Aaron_Peirsol_0001.jpg 1
.\lfw\Aaron_Peirsol\Aaron_Peirsol_0003.jpg 1
.\lfw\Aaron_Sorkin\Aaron_Sorkin_0001.jpg 1
.\lfw\Aaron_Sorkin\Aaron_Sorkin_0002.jpg 1
.\lfw\Abbas_Kiarostami\Abbas_Kiarostami_0001.jpg 0
.\lfw\Abdel_Nasser_Assidi\Abdel_Nasser_Assidi_0001.jpg 0
.\lfw\Abdoulaye_Wade\Abdoulaye_Wade_0002.jpg 1
.\lfw\Abdoulaye_Wade\Abdoulaye_Wade_0003.jpg 0
...
```

### 学習データの生成

上記の正解データを読み込み、各顔画像に対して顔検出を行い、顔領域を切り出し、笑顔ラベルとセットでファイルに保存する。

- 学習データ生成のプログラムのダウンロード
  - 以下のファイルをダウンロードし、保存する。
    - https://github.com/yusuke-morishita/lecture/blob/main/1/create_training_data.py

- 学習データ生成の実行
  - 仮想環境で、以下を実行する。
  - 学習データの`smile_dataset.pt`が生成され、保存される。
```bat
python create_training_data.py
```


## 学習の実行

### 学習１: MLP

多層パーセプトロン（Multi-layer perceptron: MLP）で、笑顔判定の画像認識モデルを学習する。

- 学習用のプログラムのダウンロード
  - 以下のファイルをダウンロードし、保存する。
    - https://github.com/yusuke-morishita/lecture/blob/main/1/train_smile_model_mlp1.py

- 画像認識モデル
  - 4層の多層パーセプトロン（Multi-layer perceptron）を用いる。
  - 認識モデルの定義は、`train_smile_model_mlp1.py`の以下の部分
```python
# Define a nuural network model (MLP)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        return self.classifier(x)
```
   - 学習パラメータは、`train_smile_model_mlp1.py`の以下の部分
```python
# training parameters
num_epochs = 10
num_batch = 64
learning_rate = 0.001
```

- 学習の実行
  - 仮想環境で、以下を実行する。
  - 学習結果の`model_weights.pth`が生成され、保存される。
```bat
python train_smile_model_mlp1.py
```

### 学習２: MLPで学習パラメータを変更

MLPを用いた学習において、学習パラメータを変更して、笑顔判定の画像認識モデルを学習する。

- 学習用のプログラムのダウンロード
  - 以下のファイルをダウンロードし、保存する。
    - https://github.com/yusuke-morishita/lecture/blob/main/1/train_smile_model_mlp1m.py

- 画像認識モデル
  - 4層の多層パーセプトロン（Multi-layer perceptron）を用いる。
  - 認識モデルの定義は、`train_smile_model_mlp1.py`と同様。
  - 学習パラメータは、`train_smile_model_mlp1m.py`の以下の部分。学習率`learning_rate`を10倍に変更。
```python
# training parameters
num_epochs = 10
num_batch = 64
learning_rate = 0.01
```

- 学習の実行
  - 仮想環境で、以下を実行する。
  - 学習結果の`model_weights.pth`が生成され、保存される。
```bat
python train_smile_model_mlp1m.py
```

### 学習３: MLPでモデル定義を変更

MLPを用いた学習において、モデル定義を変更して、笑顔判定の画像認識モデルを学習する。

- 学習用のプログラムのダウンロード
  - 以下のファイルをダウンロードし、保存する。
    - https://github.com/yusuke-morishita/lecture/blob/main/1/train_smile_model_mlp2.py

- 画像認識モデル
  - 4層の多層パーセプトロン（Multi-layer perceptron）を用いる。
  - 学習パラメータは、`train_smile_model_mlp1.py`と同様。
  - 認識モデルの定義は、`train_smile_model_mlp2.py`の以下の部分。中間層に非線形関数のReLUを追加。
```python
# Define a nuural network model (MLP)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(x)
```

- 学習の実行
  - 仮想環境で、以下を実行する。
  - 学習結果の`model_weights.pth`が生成され、保存される。
```bat
python train_smile_model_mlp.py
```

### 学習４: モデル定義をLeNetに変更

認識モデルを画像認識用に提案されたLeNetに変更して、笑顔判定の画像認識モデルを学習する。

- 学習用のプログラムのダウンロード
  - 以下のファイルをダウンロードし、保存する。
    - https://github.com/yusuke-morishita/lecture/blob/main/1/train_smile_model_lenet1.py

- 画像認識モデル
  - 1998年に画像認識用に提案されたLeNetを用いる。
    - Yann LeCun et al. Gradient Based Learning Applied to Document Recognition, 1998
  - 学習パラメータは、`train_smile_model_mlp1.py`と同様。
  - 認識モデルの定義は、`train_smile_model_lenet1.py`の以下の部分。畳み込みニューラルネットワークのConv2dなどを使用。
```python
# Define a nuural network model (LeNet)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace = True),
            nn.Linear(120, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        return self.classifier(x)
```

- 学習の実行
  - 仮想環境で、以下を実行する。
  - 学習結果の`model_weights.pth`が生成され、保存される。
```bat
python train_smile_model_lenet1.py
```

## 動作確認

### カメラを用いた動作確認: LeNet

LeNetで学習した笑顔判定の画像認識モデルを、カメラから入力した画像で動作確認する。

- 動作確認用のプログラムのダウンロード
  - 以下のファイルをダウンロードし、保存する。
    - https://github.com/yusuke-morishita/lecture/blob/main/1/test_smile_model_lenet1.py

- 動作確認の実行
  - 仮想環境で、以下を実行する。
```bat
python test_smile_model_lenet1.py
```
