from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# データ拡張の設定
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

model_path="学習用データフォルダパス"

# 訓練データの読み込み
train_generator = train_datagen.flow_from_directory(
    'model_path',#指定パス
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

# 検証データの読み込み
validation_generator = train_datagen.flow_from_directory(
    'model_path',#指定パス
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# ResNet50モデルの読み込み
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの訓練
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)

# モデルの保存
model.save("モデル保存先の絶対パス")
