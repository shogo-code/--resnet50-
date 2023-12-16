"version resnet-50"は画像ファイル分類用ディープラーニングモデルをファインチューニングで実装したものです。事前学習済みモデルとしてResnet-50を使用しています。Resnet-50を用いた学習によりミドルクラスのGPUでも動作可能であり、高い判別性能を出しています。製作者が使用するRTX3060 12GBでトレーニングすることを前提に構築されているため、手持ちのGPUのVRAMに合わせて学習率やepo数、バッチサイズを適宜変更することを推奨します。
短時間で生成できる画像が格段に増え、それらの処理が追い付かないということがよくありました。そこで自分が作りたい画像とそうでない画像、もしくは稀に生成されるクオリティが低い画像をあらかじめ分類するモデルを開発しました。個人的に欲しい画像とそうでない画像を、ある特徴をもとに分類することができます。判別基準となる”特徴”は学習に使う画像データから決まります。学習に使う画像データの手動選別はLoRAを作成する要領で行います。例えば望ましくない画像にはできるだけ無作為なものを選び、望ましい画像には特徴を絞ることを守っていれば問題なく動作します。

