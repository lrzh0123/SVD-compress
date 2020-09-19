import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



if __name__ == '__main__':
    # 读取图片
    img_eg = mpimg.imread(".../夕阳.jpg")
    print(img_eg.shape)
    F1 = img_eg.shape[0]
    F2 = img_eg.shape[1]
    F3 = img_eg.shape[2]
    # （F1, F2, F3）

    # SVD
    img_temp = img_eg.reshape(F1, F2 * F3)
    U, Sigma, VT = np.linalg.svd(img_temp)
    print(Sigma.shape)  # 667

    # 取前sval_nums个奇异值
    sval_nums = 400
    img_restruct1 = (U[:, 0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
    img_restruct1 = img_restruct1.reshape(F1, F2, F3)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(img_eg)
    ax[0].set(title="src")
    ax[1].imshow(img_restruct1.astype(np.uint8))
    ax[1].set(title="nums of sigma = " + str(sval_nums))
    plt.show()


