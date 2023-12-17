"version resnet-50"は画像ファイル分類用ディープラーニングモデルをファインチューニングで実装したものです。事前学習済みモデルとしてResnet-50を使用しています。Resnet-50を用いた学習によりミドルクラスのGPUでも動作可能であり、高い判別性能を出しています。ただし製作者が使用するRTX3060 12GBでトレーニングすることを前提に構築されているため、手持ちのGPUのVRAMに合わせて学習率やepo数、バッチサイズを適宜変更することを推奨します。

目的
画像生成が高速化することで短時間で生成できる画像が格段に増え、それらの処理が追い付かないということがよくありました。そこで自分が作りたい画像とそうでない画像、もしくは稀に生成されるクオリティが低い画像をあらかじめ分類するモデルを開発しました。利用者が個人的に欲しい画像とそうでない画像を、ある特徴をもとに分類することができます。さらにファインチューニングの行程を２段階に分けています。CI_model_pre.pyがクオリティがひどく劣るものを間引きします。CI_model_fine.pyは残る画像を特徴付けさらに分類します。今回はではA,B,Cの３つまで分類できます。これにより多様な分類のレパートリーを学習させることができます。

実行
学習用画像が保存されているフォルダのパスをプログラム内のpathに指定します。任意のフォルダで git bushを開き"git clone https://github.com/shogo-code/--version-resnet50" を実行します。CI_model_pre.pyとCI_model_fine.py、もしくはresnet,pyを実行し学習させます。CI_predict.pyで予測を実行します。参考までにpythonは3.10及び3.9.13で正常な動作が確認されています。

学習
判別基準となる”特徴”は学習に使う画像データから決まります。学習に使う画像データの手動選別はLoRAを作成する要領で行います。例えば望ましくない画像にはできるだけ無作為なものを選び、望ましい画像には特徴を絞ることを守っていれば問題なく動作します。

"version resnet-50" is a fine-tuned implementation of a deep learning model for image file classification. I am using Resnet-50 as a pretrained model. Learning using Resnet-50 allows it to operate even on middle-class GPUs and achieves high discrimination performance. However, since it is built on the premise of training on the RTX3060 12GB used by the creator, we recommend changing the learning rate, number of epo, and batch size appropriately according to the VRAM of your GPU.

Purpose
As image generation speeds up, the number of images that can be generated in a short period of time has increased significantly, and it has often been difficult to process them. Therefore, we developed a model that pre-classifies images that you want to create and images that you do not want to create, or images that are rarely generated and are of low quality. Images that users personally want and those that they don't want can be classified based on certain characteristics. Furthermore, the fine tuning process is divided into two stages. CI_model_pre.py will cull the ones with extremely poor quality. CI_model_fine.py characterizes and further classifies the remaining images. This allows it to learn a diverse repertoire of classifications.

Execution steps
Open git bush in any folder and run "git clone https://github.com/shogo-code/--version-resnet50". Run CI_model_pre.py and CI_model_fine.py or resnet,py for learning. Run the prediction with CI_predict.py. For reference, normal operation has been confirmed with python 3.10 and 3.9.13.

Training
The "features" that serve as discrimination criteria are determined from the image data used for learning. Manual selection of image data used for learning is done in the same way as creating LoRA. For example, if you choose something as random as possible for undesirable images and narrow down the features for desirable images, it will work fine.

