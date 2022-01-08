# gauge_reader
丸型メーターの値を読む

![001](https://user-images.githubusercontent.com/91955493/148638130-7bf7d12f-80ab-479f-8d1c-0043e79d1ffc.jpg)


## インストール

Python で OpenCVが動作する環境なら動作する
```
$ pip install -r requirements.txt
```

## 使用法

```
usage: gauge_reader.py [-h] [-l {warning,debug,info}] input

丸型メーターの値を読む

positional arguments:
  input

optional arguments:
  -h, --help            show this help message and exit
  -l {warning,debug,info}, --loglevel {warning,debug,info}

```

- 動作確認画像として9枚の画像を sample/ 以下に用意した
- 針が0-1000の範囲にない場合は -1 g と判定される
- -l debug を使用するとデバッグ用の画像 debug1.jpg, debug2.jpg debug3.jpg debug4.jpg を出力する

### 使用例:

```
$ python gauge_reader.py sample/gauge-1.jpg
0 g

$ python gauge_reader.py sample/gauge-2.jpg
2 g

$ python gauge_reader.py sample/gauge-3.jpg
-1 g

$ python gauge_reader.py sample/gauge-4.jpg
110 g

$ python gauge_reader.py sample/gauge-5.jpg
378 g

$ python gauge_reader.py sample/gauge-6.jpg
869 g

$ python gauge_reader.py sample/gauge-7.jpg
423 g

$ python gauge_reader.py sample/gauge-8.jpg
-1 g

$ python gauge_reader.py sample/gauge-9.jpg
616 g
```

## コメント

同一のアナログメーターが複数の数値を示す画像をネット上に見つけることができなかったので100円ショップで[キッチン目安計](https://watts-online.jp/products/5470)なるものを購入して、スマホで撮影して使用した

この製品は検定を受けたものではない「なんちゃって計り」なので、針がメモリに届かない範囲があるなどイマイチな製品ではあるが今どきの一般家庭にアナログメーターで自由に数値を指定できるようなものがあるはずもなく仕方なく代用した

### 仕組み

このプログラムが数値を計測する仕組みは単純で次のようなことを行っている

1. 針の方向をOpenCVの領域抽出を使用して計算
2. 針の回転中心から0の目盛り方向のベクトルと針の回転中心から針の方向のベクトルから針の角度を計算
3. 計算した角度を目盛りの数値に変換

![1](https://user-images.githubusercontent.com/91955493/148638058-871e5c60-1db1-4dd4-a6ac-d733fd7d9219.jpg)
![4](https://user-images.githubusercontent.com/91955493/148638060-e9ff8f55-88c9-4ca5-a827-be3d36d5610c.jpg)
![7](https://user-images.githubusercontent.com/91955493/148638061-87eea91b-fc90-470e-b8f3-b19b81df02b1.jpg)

### P.S.

110円とはいえ経費持ち出しになる課題は勘弁してください……
