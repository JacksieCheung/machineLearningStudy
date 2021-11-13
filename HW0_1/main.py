import numpy as np


def main():
    a = np.loadtxt('martixA.txt')
    b = np.loadtxt('martixB.txt')
    c = a.dot(b)
    print(c)

    # axis=0 按列排序，axis=1 按行排序，没有 axis 默认按行排序
    c.sort()
    # print(c)
    np.savetxt("./ans_one.txt", c, fmt='%d', delimiter=' ')
    print('finished')


if __name__ == '__main__':
    main()
