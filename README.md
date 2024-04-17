## resnet-50  
"version resnet-50"は画像ファイル分類用ディープラーニングモデルをファインチューニングで実装したものです。事前学習済みモデルとしてResnet-50を使用しています。Resnet-50を用いた学習によりミドルクラスのGPUで動作可能であり、高い判別性能を出しています。12GB GPUでトレーニングすることを前提に構築されています（実行はいわずもがな）。  

画像生成は高速化してきました。短時間で生成できる画像数が格段に増えるとそれに伴う破綻した画像も増加し、量的処理も追い付かなくなりました。そこで自分が作りたい画像と稀に生成されるクオリティが低い画像をあらかじめ分類するモデルを設計しました。さらに利用者が個人的に欲しい画像とそうでない画像を、ある特徴をもとに分類することもできます。  
CI_model_pre.py；クオリティがひどく劣るものを間引きし、残る画像を特徴付けます。(つまりそういう風に仕分けを学習させます)  
CI_model_fine.py；画像をさらに分類します。a,b,cとなっているのはタグですので書き換え自由です。  
CI_pridict.py；学習後に実行して予測させます。同時に指定フォルダに保存させます。このモデルではA,B,Cの３つまで分類できます。分類数を増やすほどより多様な分類を学習させることができますが、精度の低下を伴う可能性も高まります。  
resnet.py；上２つをにまとめたものです。ファインチューニングにresnet-50を使用しています。学習後はCI_pridict.pyで予測します。  


学習用データを用意し任意のフォルダに保存します。学習用画像が保存されているフォルダのパスをプログラム内のpathに指定します。python環境(VScodeなど)でCI_model_preとCI_model_fineの２つ、もしくはresnetのみを用いて実行し学習させます。前者はまずCI_model_fine.pyで特徴分類します。その後CI_predict.pyで予測を実行します。予測された画像は指定のフォルダに保存されます。pythonは3.10及び3.9.13で正常な動作が確認されています。リアルな画像ほどresnet.pyでの学習に向いています。ただし絵やキャラクターなどには適していません。その場合はCI_model_pre.pyとCI_model_fine.pyにより分類するといった具合です。

判別基準となる”特徴”は学習に使う画像データから決まります。学習に使う画像データの手動選別はLoRAを作成する要領で行います。例えば望ましくない画像にはできるだけ無作為なものを選び、望ましい画像には特徴を絞ることを守っていれば問題なく動作します。

epoc=10は製作者使用のRTX3060における一定の動作限界です。お使いのGPUに合わせて変更してください。

"version resnet-50" is a fine-tuned implementation of a deep learning model for image file classification. I am using Resnet-50 as a pretrained model. Learning using Resnet-50 allows it to operate even on middle-class GPUs and achieves high discrimination performance. However, since it is built on the premise of training on the RTX3060 12GB used by the creator, we recommend changing the learning rate, number of epo, and batch size appropriately according to the VRAM of your GPU.

Purpose
As image generation speeds up, the number of images that can be generated in a short period of time has increased significantly, and it has often been difficult to process them. Therefore, we developed a model that pre-classifies images that you want to create and images that you do not want to create, or images that are rarely generated and are of low quality. Images that users personally want and those that they don't want can be classified based on certain characteristics. Furthermore, the fine tuning process is divided into two stages. CI_model_pre.py will cull the ones with extremely poor quality. CI_model_fine.py characterizes and further classifies the remaining images. This allows it to learn a diverse repertoire of classifications.

Execution steps
Open git bush in any folder and run "git clone https://github.com/shogo-code/--version-resnet50". Run CI_model_pre.py and CI_model_fine.py or resnet,py for learning. Run the prediction with CI_predict.py. For reference, normal operation has been confirmed with python 3.10 and 3.9.13.

Training
The "features" that serve as discrimination criteria are determined from the image data used for learning. Manual selection of image data used for learning is done in the same way as creating LoRA. For example, if you choose something as random as possible for undesirable images and narrow down the features for desirable images, it will work fine.

