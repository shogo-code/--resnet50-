import yaml
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# configファイルの読み込み
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# データディレクトリのパス
train_data_dir = config['data_directory']

# 画像の前処理
train_datagen = ImageDataGenerator(
    rescale=config['image_processing']['rescale'],
    zoom_range=tuple(config['image_processing']['zoom_range']),
    validation_split=config['image_processing']['validation_split']
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=config['training_parameters']['batch_size'],
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=config['training_parameters']['batch_size'],
    class_mode='binary',
    subset='validation'
)

# ResNet-50モデルのロード
base_model = ResNet50(weights='imagenet', include_top=False)

# カスタム出力層の追加
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(config['model_parameters']['dropout_rate'])(x)  # ドロップアウト層の追加
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ResNetの層を凍結
for layer in base_model.layers:
    layer.trainable = False

# コンパイル
model.compile(optimizer=Adam(learning_rate=config['model_parameters']['learning_rate']),
              loss='binary_crossentropy', metrics=['accuracy'])

# ファインチューニング
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config['training_parameters']['epochs'],
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
model.save('C:\\Users\\shogo11\\Desktop\\mig\\sresult\\my_zoomin6_model') # モデル保存先パス

