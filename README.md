# Lecture

## Python仮想環境の準備

フォルダの作成
```bat
mkdir lecture1
cd lecture1
```

仮想環境の作成
```bat
REM Python公式の場合
python -m venv .venv

REM Anacondaの場合
conda create -n .venv python
```

仮想環境の起動（Python公式の場合）
```bat
REM Python公式の場合
.venv\Scripts\activate

REM Anacondaの場合
conda activate .venv
```
