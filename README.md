# Tabular Data Augmentation
## 環境
- python 3.8.10
- wsl

## 環境構築
```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## 実行コマンド
```bash
# augmenters通りにDAされる
python main.py

## augmentersの全組み合わせが実行される
python main_all.py
```

## Config.yaml
以下をいじることでaugmentするアルゴリズムを切り替えられる。
```yaml
augmenters: [random_flip,
             noise, 
             random_collapse, 
             shuffle, 
             random_resize
             ]
```
例えば、`noise`だけかけたい時は以下のようにする。
```yaml
augmenters: [noise]
```
 