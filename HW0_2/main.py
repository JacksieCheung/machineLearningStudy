# PIL 是 python 的图像库
from PIL import Image


def main():
    lena = Image.open("lena.png")
    lena_modified = Image.open("lena_modified.png")

    # size 获取大小
    w, h = lena.size
    for j in range(h):
        for i in range(w):
            # 遍历比较像素点，其实就是个矩阵，一样的就变成透明
            if lena.getpixel((i, j)) == lena_modified.getpixel((i, j)):
                # putpixel 方法，改变像素点，第二个参数是颜色
                # lena_modified.putpixel((i, j), 255)
                lena_modified.putpixel((i, j), 0)

    # 把图片展示出来
    lena_modified.show()
    lena_modified.save("ans_two.png")
    print('finished')


if __name__ == '__main__':
    main()