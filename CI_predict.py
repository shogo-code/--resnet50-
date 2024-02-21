from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# モデルの読み込み
pre_model_path = "C:/Users/shogo11/Desktop/mig/Hanbetsu/Pre_model.tf"
fine_model_path = "C:/Users/shogo11/Desktop/mig/Hanbetsu/Fine_model.tf"
pre_model = load_model(pre_model_path)
fine_model = load_model(fine_model_path)

# 予測対象のフォルダ
folder_path = '予測したい画像が含まれるフォルダのパス'  

# カテゴリ
fine_categories_jp = ['a','b','c']

# フォルダ内のすべての画像を予測
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # 第1段階の予測: "high-quality"と"other-quality"
    pre_predictions = pre_model.predict(x)
    if pre_predictions[0][0] > 0.5:
        # "high-quality"と判定された場合、さらに細かい分類を行う
        fine_predictions = fine_model.predict(x)
        predicted_class = np.argmax(fine_predictions[0])
        print(f"画像 {filename}: 予測結果: 高レベルの{fine_categories_jp[predicted_class]}")
    else:
        print(f"画像 {filename}: 予測結果: その他")
