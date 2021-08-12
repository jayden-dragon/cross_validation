from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

def deprocess_image(X):
    X -= X.mean()
    X /= (X.std() + 1e-5)
    X *= 0.1
    X += 0.5
    X = np.clip(X, 0, 1)
    X *= 255
    X = np.clip(X, 0, 255).astype('uint8')
    return X


def draw_activation(activation, figure_name):
 images_per_row = 16
 n_features = activation.shape[-1]
 size = activation.shape[1]
 n_cols = n_features // images_per_row
 display_grid = np.zeros((size * n_cols, images_per_row * size))
 for col in range(n_cols):
    for row in range(images_per_row):
        channel_image = activation[0, :, :, col * images_per_row + row]
        channel_image = deprocess_image(channel_image)
        display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
 scale = 1. / size
 plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
 plt.title(figure_name)
 plt.grid(False)
 plt.imshow(display_grid, aspect='auto', cmap='viridis')
 plt.show()


model = VGG16(weights='imagenet')
#model = load_model('hw3_q2_VGG16_2.h5') #load previous model
model.summary()

def generate_patterns(layer_name, filter_index, size = 150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads=K.gradients(loss, model.input)[0]
    grads/=(K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1,size,size,3))* +128
    step = 1
    for i in range(40):
     loss_value, grads_value = iterate([input_img_data])
     input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

def draw_filters(layer_name, size = 150):
   images_per_row = 8
   n_cols = 8
   results = np.zeros((size*n_cols, images_per_row*size, 3), dtype = 'uint8')
   for row in range(n_cols):
      for col in range(images_per_row):
          print((col,row))
          filter_image = generate_patterns(layer_name, col+row*8, size = size)
          results[col*size:(col+1)*size,
          row*size:(row+1)*size, :] = filter_image
   plt.figure(figsize = (20,20))
   plt.imshow(results)
   plt.title(layer_name)
   plt.show()
   return filter_image, results

def gradCAM(model, x):
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    max_output = model.output[:, np.argmax(preds)]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(max_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap, conv_layer_output_value, pooled_grads_value

layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
for layer_name in layer_names:
   print(layer_name)
   filter_image, results = draw_filters(layer_name)

img_path = './chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)

heatmap, conv_output, pooled_grads = gradCAM(model, img_tensor)

import cv2
img=cv2.imread(img_path)
heatmap=cv2.resize(heatmap, (img.shape[1],img.shape[0]))
heatmap=np.uint8(255*heatmap)
heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img=heatmap*0.4+img
cv2.imwrite('hw3_6.jpg', superimposed_img)

draw_no = range(256,256+32,1)
conv_activation=np.expand_dims(conv_output[:,:,draw_no], axis=0)
draw_activation(conv_activation, 'last_conv')
plt.matshow(pooled_grads[draw_no].reshape(-1,16), cmap='viridis')

