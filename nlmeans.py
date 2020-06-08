import scipy
import math
import time
from tqdm import *

def window(x, y, r, img, k):
    # x,y为中心 定下以r为半径的窗口
    # 值传入？
    ro,co,ch = img.shape
    p = [x,y]
    mtx = []
    for i in range(-r,r+1):
        for j in range(-r,r+1):
            if x+i<0 or y+j<0 or x+i>=ro or y+j>=co:
                mtx.append(-1)
            else:
                mtx.append(img[x+i][y+j][k])
    return mtx
    # 如果越界处理办法？

def eu_dist(m1,m2):
    if len(m1)!=len(m2):
        print("len dif")
        return 0
    else:
        t = 0
        cnt = 0
        for i in range(len(m1)):
            if m2[i]==-1 or m1[i]==-1:
                t += 0
            else:
                t += m1[i]-m1[i]
                cnt += 1
    t = t*t
    if cnt!=0:
        t = t/cnt
    else:
        t=-998
        print("cnt=0")
    return t

def nl_means(img, ad, se, p):
    h2 = p*2
    h2=256
    co,ro,ch = img.shape
    ab=1
    t_img = np.copy(img)
    mask = get_noise_mask(img)
    for i in tqdm(range(co),ncols=80):
        for j in range(ro):
            for k in range(ch):
                if mask[i][j][k]==-1:
                    continue
                #if ab==1:
                else:
                    #定点了 i,j,k
                    ct = [i,j,k]
                    mtc = window(i,j,ad,img,k)
                    csum=0
                    cava=0
                    dstm=0
                    
                    for a in range(max(0,i-se),min(co,i+se)):
                        for b in range(max(0,j-se),min(ro,j+se)):
                            if a==i and b==j or t_img[a][b][k]==0:
                            #if a==i and b==j:
                                continue
                            mtt=window(a,b,ad,img,k)
                            dst = eu_dist(mtc,mtt)
                            dst = math.exp((-1)*(dst)/h2)
                            if dst>dstm:
                                dstm=dst
                            csum += dst
                            cava += dst*t_img[a][b][k]
                    csum += dstm
                    cava += dstm * t_img[i][j][k]
                    if csum!=0:
                        t_img[i][j][k] = cava/csum
                    else:
                        t_img[i][j][k] = img[i][j][k]
    return t_img
def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)
    stddif=noise_mask.std()
    # -------------实现图像恢复代码答题区域----------------------------
    res_img = nl_means(noise_img,1,4,stddif)
    # ---------------------------------------------------------------

    return res_img
