import numpy as np
import cv2
import matplotlib.pyplot as plt

#Read images in and convert to grayscale
image1a = cv2.imread("frame1_a.png")
image1a = cv2.cvtColor(image1a, cv2.COLOR_BGR2GRAY)
image1b = cv2.imread("frame1_b.png")
image1b = cv2.cvtColor(image1b, cv2.COLOR_BGR2GRAY)
image2a = cv2.imread("frame2_a.png")
image2a = cv2.cvtColor(image2a, cv2.COLOR_BGR2GRAY)
image2b = cv2.imread("frame2_b.png")
image2b = cv2.cvtColor(image2b, cv2.COLOR_BGR2GRAY)


#Storing Gaussian Filters
DoGgx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
DoGgy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

#Convolution Function
def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    image_row, image_col = output.shape
    padded_output = np.zeros((image_row + (2), image_col + (2)))
    padded_output[1:padded_output.shape[0] - 1, 1:padded_output.shape[1] - 1] = output

    return padded_output

#Optical Flow Function
def flow(imagea, imageb, imgx, imgy, xy):
    x = xy[0]
    y = xy[1]
    A = np.array([[imgx[x-1][y-1] , imgy[x-1][y-1]],
                 [imgx[x][y-1] , imgy[x][y-1]],
                 [imgx[x+1][y-1] , imgy[x+1][y-1]],
                 [imgx[x-1][y] , imgy[x-1][y]],
                 [imgx[x][y] , imgy[x][y]],
                 [imgx[x+1][y] , imgy[x+1][y]],
                 [imgx[x-1][y+1] , imgy[x-1][y+1]],
                 [imgx[x][y+1] , imgy[x][y+1]],
                 [imgx[x+1][y+1] , imgy[x+1][y+1]]])
    b = np.array([
        [int(imageb[x-1][y-1]) - int(imagea[x-1][y-1])],
        [int(imageb[x][y-1]) - int(imagea[x][y-1])],
        [int(imageb[x+1][y-1]) - int(imagea[x+1][y-1])],
        [int(imageb[x-1][y]) - int(imagea[x-1][y])],
        [int(imageb[x][y]) - (imagea[x][y])],
        [int(imageb[x+1][y]) - int(imagea[x+1][y])],
        [int(imageb[x-1][y+1]) - int(imagea[x-1][y+1])],
        [int(imageb[x][y+1]) - int(imagea[x][y+1])],
        [int(imageb[x+1][y+1]) - int(imagea[x+1][y+1])]
    ])
    b = -b
    AT = A.transpose()
    ATA = np.matmul(AT,A)
    ATAinv = np.linalg.pinv(ATA)
    ATb = np.matmul(AT,b)
    flo = np.matmul(ATAinv,ATb)
    return np.array([flo[0][0], flo[1][0]])


#Image 1 Convolutions
image1agx = convolution(image1a, DoGgx)
image1agy = convolution(image1a, DoGgy)
#Image 2 Convolutions
image2agx = convolution(image2a, DoGgx)
image2agy = convolution(image2a, DoGgy)


image_row, image_col = image1a.shape
image1flowvx = np.zeros(image1a.shape)
image1flowvy = np.zeros(image1a.shape)
for row in range(image_row-1):
    for col in range(image_col-1):
        print("Image 1 ",row,"/",image_row,"   ",col,"/",image_col)
        a = flow(image1a, image1b, image1agx, image1agy, [row,col])
        image1flowvx[row,col] = a[0]
        image1flowvy[row,col] = a[1]

image_row, image_col = image2a.shape
image2flowvx = np.zeros(image2a.shape)
image2flowvy = np.zeros(image2a.shape)
for row in range(image_row-1):
    for col in range(image_col-1):
        print("Image 2 ",row,"/",image_row,"   ",col,"/",image_col)
        a = flow(image2a, image2b, image2agx, image2agy, [row,col])
        image2flowvx[row,col] = a[0]
        image2flowvy[row,col] = a[1]


vxvy1 = np.sqrt(image1flowvx**2 + image1flowvy**2)
vxvy2 = np.sqrt(image2flowvx**2 + image2flowvy**2)

#Image 1 visuals
plt.imshow(image1flowvx, cmap='gray', vmin=0, vmax=1)
plt.title("Image 1, V_x")
plt.show()
plt.clf()
plt.imshow(image1flowvy, cmap='gray', vmin=0, vmax=1)
plt.title("Image 1, V_y")
plt.show()
plt.clf()
plt.imshow(vxvy1, cmap='gray', vmin=0, vmax=1)
plt.title("sqrt( V_x^2 + V_y^2 )")
plt.show()
plt.clf()
#Image 2 visuals
plt.imshow(image2flowvx, cmap='gray', vmin=0, vmax=1)
plt.title("Image 2, V_x")
plt.show()
plt.clf()
plt.imshow(image2flowvy, cmap='gray', vmin=0, vmax=1)
plt.title("Image 2, V_y")
plt.show()
plt.clf()
plt.imshow(vxvy2, cmap='gray', vmin=0, vmax=1)
plt.title("sqrt( V_x^2 + V_y^2 )")
plt.show()
plt.clf()
