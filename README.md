##installation##
open your terminal and run it
`pip install -r requirements.txt`


"version resnet-50" is a fine-tuned implementation of a deep learning model for image file classification. I am using Resnet-50 as a pretrained model. Learning using Resnet-50 allows it to operate even on middle-class GPUs and achieves high discrimination performance. However, since it is built on the premise of training on the RTX3060 12GB used by the creator, we recommend changing the learning rate, number of epo, and batch size appropriately according to the VRAM of your GPU.

Purpose
As image generation speeds up, the number of images that can be generated in a short period of time has increased significantly, and it has often been difficult to process them. Therefore, we developed a model that pre-classifies images that you want to create and images that you do not want to create, or images that are rarely generated and are of low quality. Images that users personally want and those that they don't want can be classified based on certain characteristics. Furthermore, the fine tuning process is divided into two stages. CI_model_pre.py will cull the ones with extremely poor quality. CI_model_fine.py characterizes and further classifies the remaining images. This allows it to learn a diverse repertoire of classifications.

Execution steps
Open git bush in any folder and run "git clone https://github.com/shogo-code/--version-resnet50". Run CI_model_pre.py and CI_model_fine.py or resnet,py for learning. Run the prediction with CI_predict.py. For reference, normal operation has been confirmed with python 3.10 and 3.9.13.

Training
The "features" that serve as discrimination criteria are determined from the image data used for learning. Manual selection of image data used for learning is done in the same way as creating LoRA. For example, if you choose something as random as possible for undesirable images and narrow down the features for desirable images, it will work fine.

