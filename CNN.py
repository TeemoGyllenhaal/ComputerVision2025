# %%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Tải dữ liệu
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Xây dựng mô hình CNN
model = models.Sequential([
    # Lớp tích chập
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    # Lớp tổng hợp
    layers.MaxPooling2D((2, 2)),
    # Lớp tích chập thứ hai
    layers.Conv2D(64, (3,3), activation='relu'),
    # Lớp tổng hợp thứ hai
    layers.MaxPooling2D((2, 2)),
    # Chuyển thành vector
    layers.Flatten(),
    # Lớp kết nối đầy đủ
    layers.Dense(128, activation='relu'),
    # Lớp đầu ra (phân loại 10 lớp)
    layers.Dense(10, activation='softmax')
])

# 3. Biên dịch và huấn luyện
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 4. Đánh giá
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Độ chính xác trên tập kiểm tra: {test_acc*100:.2f}%")

#-------------------------------Thử nghiệm với ảnh --------------------------------#
# %%
import numpy as np
import matplotlib.pyplot as plt

# Dự đoán xác suất các lớp
predictions = model.predict(x_test)
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
# Hàm tìm lớp có xác suất cao nhất
def predict_label(i):
    pred = np.argmax(predictions[i])
    true = y_test[i]
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Dự đoán: {class_names[pred]} - Thực tế: {class_names[true]}")
    plt.axis('off')
    plt.show()

# Thử 5 ảnh đầu tiên
for i in range(5):
    predict_label(i)
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.show()


# %%
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('img/test.jpg', target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0
img_array = 1 - img_array  # đảo màu nền trắng thành đen, giống Fashion-MNIST
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array) 
pred_label = np.argmax(prediction) # Danh sách class (Fashion-MNIST) 

class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ] # Hiển thị kết quả 
plt.imshow(img, cmap='gray') # <-- dùng 'img' gốc (PIL Image)

plt.title(f"Dự đoán: {class_names[pred_label]}")
plt.axis('off')
plt.show()


# %%
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_custom_image(path):
    img = image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = 1 - img_array
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction)

    plt.imshow(img, cmap='gray')
    plt.title(f"Dự đoán: {class_names[pred_label]}")
    plt.axis('off')
    plt.show()

    for i, name in enumerate(class_names):
        print(f"{name:15s}: {prediction[0][i]*100:.2f}%")
# %%
# Gọi hàm:
predict_custom_image(r'img/test8.jpg')
predict_custom_image('img/test1.jpg')
predict_custom_image('img/test2.jpg')
predict_custom_image('img/test4.jpg')
predict_custom_image('img/test3.png')
predict_custom_image('img/test5.jpg')
predict_custom_image('img/test6.jpg')
predict_custom_image('img/test7.jpg')


# %%
