import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
img1 = cv2.imread('IM-0001-0001.jpeg')
img2 = cv2.imread('person1_virus_6.jpeg')

print(img1)
print(img2)

fig = plt.figure()
rows = 1
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(img1)
ax1.set_title('Normal')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(img2)
ax2.set_title('Pneumonia')
ax2.axis("off")

plt.show()