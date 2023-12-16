from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# データ拡張の設定
train_datagen_fine = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

model_path="学習に使うデータのパス"

# 訓練データの読み込み
train_generator_fine = train_datagen_fine.flow_from_directory(
    'model_path',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')

# 検証データの読み込み
validation_generator_fine = train_datagen_fine.flow_from_directory(
    'model_path',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# ResNet50モデルの読み込み
base_model_fine = ResNet50(weights='imagenet', include_top=False)
x_fine = base_model_fine.output
x_fine = GlobalAveragePooling2D()(x_fine)
x_fine = Dense(1024, activation='relu')(x_fine)
predictions_fine = Dense(3, activation='softmax')(x_fine)

model_fine = Model(inputs=base_model_fine.input, outputs=predictions_fine)

# モデルのコンパイル
model_fine.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練
model_fine.fit(
    train_generator_fine,
    epochs=10,
    validation_data=validation_generator_fine)

# モデルの保存
model_fine.save("モデル保存先前提絶対パス")
