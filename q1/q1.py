import numpy as np
import matplotlib.pyplot as plt


def getLine(F_matrix, Point):
    return np.matmul(F_matrix,np.c_[Point, 1].T)

# initializations

Pts_image1 = np.mat([[381,402],[452, 497],[671, 538],[501, 254],[506, 381], [474, 440],[471, 537],[498, 364],[706, 319],[635, 367]])
Pts_image2 = np.mat([[390, 346],[439, 412],[651, 417],[477, 194],[482, 300],[456, 359],[454, 444],[475, 287],[686, 185],[606, 253]])
F_matrix = np.mat([[-1.29750186e-06,  8.07894025e-07,  1.84071967e-03],[3.54098411e-06,  1.05620725e-06, -8.90168709e-03],[-3.29878312e-03,  5.14822628e-03,  1.00000000e+00]])


# ans = getLine(F_matrix,Pts_image1[0,:])
# print(ans.shape)

x = np.linspace(0,1242)
plt.subplot(121)

for i in range(Pts_image2.shape[0]):
    Line2 = getLine(F_matrix.T,Pts_image2[i,:])
    y2 = - (Line2[0,0]*x + Line2[2,0]) / Line2[1,0]
    plt.plot(x,y2,'g-')
    plt.plot(Pts_image1[i,0],Pts_image1[i,1],'ro')
    
image1 = plt.imread('img1.jpg')
plt.imshow(image1)

plt.subplot(122)
for i in range(Pts_image1.shape[0]):

    Line = getLine(F_matrix,Pts_image1[i,:])
    y = - (Line[0,0]*x + Line[2,0]) / Line[1,0]
    plt.plot(x,y,'g-')
    plt.plot(Pts_image2[i,0],Pts_image2[i,1],'ro')

image2 = plt.imread('img2.jpg')
plt.imshow(image2)
plt.show()    