import tkinter
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
# importing required libraries of opencv
import cv2
# importing library for plotting
from matplotlib import pyplot as plt
import numpy
import math
import numpy as np

# Define the main program parametrs

root = tkinter.Tk()
root.geometry("800x240")
root.configure(background='white')
img_cv = None  # here is going to be image


# This function is responsible for loading monochromatic image
def load_mono():
    global img_cv
    if img_cv is None:
        cv2.destroyWindow(winname="Original Image")
        img_cv = None

    root.filename = filedialog.askopenfilename(initialdir="./", title="Wybierz plik", filetypes=(
    ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    if root.filename is None:
        return

    img_cv = cv2.imread(root.filename, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow(winname="Original Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Original Image", mat=img_cv)


def load_color():
    global img_cv
    if img_cv is None:
        cv2.destroyWindow(winname="Original Image")
        img_cv = None

    root.filename = filedialog.askopenfilename(initialdir="./", title="Wybierz plik", filetypes=(
    ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    if root.filename is None:
        return

    img_cv = cv2.imread(root.filename)
    cv2.namedWindow(winname="Original Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Original Image", mat=img_cv)


def histogram():
    channels = cv2.split(img_cv)
    colors = ("b", "g", "r")

    # create the histogram plot, with three lines, one for
    # each color
    for (channel, c) in zip(channels, colors):
        histr = numpy.zeros(shape=(256))
        h = img_cv.shape[0]
        w = img_cv.shape[1]

        # loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):
                histr[channel[y, x]] += 1

        plt.bar(numpy.arange(256), histr, color=c)

    plt.xlim([0, 256])
    plt.xlabel("Color value")
    plt.ylabel("Pixels")
    plt.show()


def negacja():
    img2 = cv2.bitwise_not(img_cv)

    cv2.namedWindow(winname="Negacja", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Negacja", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def progowanie():
    prog = int(input("Please enter threshold value : "))
    print(prog)

    h = img_cv.shape[0]
    w = img_cv.shape[1]

    image_prog = img_cv.copy()
    for y in range(0, h):
        for x in range(0, w):
            image_prog[y, x] = 0 if image_prog[y, x] <= prog else 255

    cv2.namedWindow(winname="Threshold Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Threshold Image", mat=image_prog)

    plt.hist(image_prog.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def progowanie_z_zachowaniem():
    prog = int(input("Please enter threshold value : "))
    print(prog)

    h = img_cv.shape[0]
    w = img_cv.shape[1]

    image_prog = img_cv.copy()
    for y in range(0, h):
        for x in range(0, w):
            image_prog[y, x] = image_prog[y, x] if image_prog[y, x] <= prog else 255

    cv2.namedWindow(winname="Threshold Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Threshold Image", mat=image_prog)

    plt.hist(image_prog.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def rozciaganie():
    hist, bins = numpy.histogram(img_cv.flatten(), 256, [0, 256])
    h = img_cv.shape[0]
    w = img_cv.shape[1]

    P1 = 255
    P2 = 0
    for y in range(0, h):
        for x in range(0, w):
            P1 = min(P1, img_cv[y, x])
            P2 = max(P2, img_cv[y, x])
    Q3 = 0
    Q4 = 255
    img2 = img_cv.copy()

    for y in range(0, h):
        for x in range(0, w):
            if img2[y, x] <= P1:
                img2[y, x] = Q3
            elif img2[y, x] >= P2:
                img2[y, x] = Q4
            else:
                img2[y, x] = (img2[y, x] - P1) * Q4 / (P2 - P1)
    cv2.namedWindow(winname="Streching", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Streching", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def equalizacja():
    h = img_cv.shape[0]
    w = img_cv.shape[1]
    N = h * w
    nk = numpy.zeros(shape=(256))

    for y in range(0, h):
        for x in range(0, w):
            nk[img_cv[y, x]] += 1

    hist = numpy.zeros(shape=(256))
    img2 = img_cv.copy()
    new_val = numpy.zeros(shape=(256))

    poziom_s = 0.0
    for y in range(0, 255):
        poziom_s += nk[y] / N
        val = int(poziom_s // (1.0 / 256.0))
        hist[val] += nk[y]
        new_val[y] = val

    for y in range(0, h):
        for x in range(0, w):
            img2[y, x] = new_val[img2[y, x]]

    cv2.namedWindow(winname="Eq", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Eq", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def rozciaganie_param():
    P1 = int(input("Please enter P1 (min): "))
    P2 = int(input("Please enter P2 (max): "))
    Q3 = int(input("Please enter Q3 (L min): "))
    Q4 = int(input("Please enter Q4 (L max): "))

    h = img_cv.shape[0]
    w = img_cv.shape[1]

    img2 = img_cv.copy()

    for y in range(0, h):
        for x in range(0, w):
            if img2[y, x] <= P1:
                img2[y, x] = Q3
            elif img2[y, x] >= P2:
                img2[y, x] = Q4
            else:
                img2[y, x] = (img2[y, x] - P1) * Q4 / (P2 - P1)

    hist, bins = numpy.histogram(img2.flatten(), 256, [0, 256])

    cv2.namedWindow(winname="Streching", flags=cv2.WINDOW_NORMAL)
    # cv2.imwrite('rozciaganie.jpg', img2)
    cv2.imshow(winname="Streching", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def poziomy_szar():
    M = int(input("Please enter number of gray levels(M) : "))
    level = 255 / M
    val = 255 / (M - 1)

    h = img_cv.shape[0]
    w = img_cv.shape[1]

    image_prog = img_cv.copy()
    for y in range(0, h):
        for x in range(0, w):
            image_prog[y, x] = int(image_prog[y, x] // level * val)

    cv2.namedWindow(winname="Graysacle reduction Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Graysacle reduction Image", mat=image_prog)

    plt.hist(image_prog.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def mediana(mask):
    if (mask % 2 == 0 or mask < 3):
        return

    img2 = cv2.medianBlur(img_cv, mask)

    cv2.namedWindow(winname="Median", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Median", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def mediana_win():
    win = tkinter.Toplevel()
    win.wm_title("Median")

    l = tkinter.Scale(win, from_=3, to=11, tickinterval=2, orient=tkinter.HORIZONTAL, label="Mask's size")
    l.grid(row=0, column=0)

    b = ttk.Button(win, text="Okay", command=lambda: mediana(l.get()))
    b.grid(row=1, column=0, pady=20, padx=30)
    return


def filter(tab):
    img2 = cv2.filter2D(img_cv, -1, tab)

    cv2.namedWindow(winname="Filter", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Filter", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def laplacian(tab):
    imgl = cv2.filter2D(img_cv, -1, tab)

    cv2.namedWindow(winname="Laplacian", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Laplacian", mat=imgl)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(imgl.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def smoothing_win():
    smooth1 = numpy.array((
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]), dtype="float") / 8
    smooth2 = numpy.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]), dtype="float") / 9
    smooth3 = numpy.array((
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]), dtype="float") / 16

    win = tkinter.Toplevel()
    win.wm_title("Smothing")

    b1 = ttk.Button(win, text="[0, -1, 0]\n[-1, 4, -1]\n[0, -1, 0]", command=lambda: filter(smooth1))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[-1, -1, -1]\n[-1, 8, -1]\n[-1, -1, -1]", command=lambda: filter(smooth2))
    b2.grid(row=0, column=1, pady=20, padx=30)
    b3 = ttk.Button(win, text="[1, 2, 1]\n[2, 4, 2]\n[1, 2, 1]", command=lambda: filter(smooth3))
    b3.grid(row=0, column=2, pady=20, padx=30)
    b4 = ttk.Button(win, text="[1, 1, 1]\n[1, K, 1]\n[1, 1, 1]", command=lambda: filter(
        numpy.array((
            [1, 1, 1],
            [1, int(b6.get()), 1],
            [1, 1, 1]), dtype="float") / (8 + int(b6.get()))
    ))
    b4.grid(row=1, column=0, pady=20, padx=30)
    b5 = ttk.Button(win, text="[0, 1, 0]\n[1, K, -1]\n[0, 1, 0]", command=lambda: filter(
        numpy.array((
            [0, 1, 0],
            [1, int(b6.get()), 1],
            [0, 1, 0]), dtype="float") / (4 + int(b6.get()))
    )
                    )
    b5.grid(row=1, column=1, pady=20, padx=30)

    b6 = ttk.Label(win, text="K = ")
    b6.grid(row=1, column=4, pady=20)
    b6 = ttk.Entry(win)
    b6.grid(row=1, column=5, pady=20)


def laplacian_win():
    sharpen1 = numpy.array((
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]), dtype="int")
    sharpen2 = numpy.array((
        [1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]), dtype="int")
    sharpen3 = numpy.array((
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]), dtype="int")

    win = tkinter.Toplevel()
    win.wm_title("Laplacian")

    b1 = ttk.Button(win, text="[0, -1, 0]\n[-1, 4, -1]\n[0, -1, 0]", command=lambda: laplacian(sharpen1))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[-1, -1, -1]\n[-1, 8, -1]\n[-1, -1, -1]", command=lambda: laplacian(sharpen2))
    b2.grid(row=0, column=1, pady=20, padx=30)
    b3 = ttk.Button(win, text="[1, -2, 1]\n[-2, 4, -2]\n[1, -2, 1]", command=lambda: laplacian(sharpen3))
    b3.grid(row=0, column=2, pady=20, padx=30)


def detect(tab):
    img2 = cv2.filter2D(img_cv, -1, tab)

    cv2.namedWindow(winname="Detect", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Detect", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def detection_win():
    detect1 = numpy.array((
        [1, -2, 1],
        [-2, 5, -2],
        [1, -2, 1]), dtype="int")
    detect1 = numpy.array((
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]), dtype="int")
    detect3 = numpy.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")

    win = tkinter.Toplevel()
    win.wm_title("Detection")

    b1 = ttk.Button(win, text="[1, -2, 1]\n[-2, 5, -2]\n[1, -2, 1]", command=lambda: detect(detect1))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[-1, -1, -1]\n[-1, 9, -1]\n[-1, -1, -1]", command=lambda: detect(detect1))
    b2.grid(row=0, column=1, pady=20, padx=30)
    b3 = ttk.Button(win, text="[0, -1, 0]\n[-1, 5 ,-1]\n[0, -1, 0]", command=lambda: detect(detect3))
    b3.grid(row=0, column=2, pady=20, padx=30)


def sobel_x():
    # construct the Sobel x-axis kernel
    sobelX = numpy.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")

    img2 = cv2.filter2D(img_cv, -1, sobelX)

    cv2.namedWindow(winname="Sobel X", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Sobel X", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def sobel_y():
    # construct the Sobel y-axis kernel
    sobelY = numpy.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

    img2 = cv2.filter2D(img_cv, -1, sobelY)

    cv2.namedWindow(winname="Sobel Y", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Sobel Y", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def prewitt_x():
    # construct the Prewitt x-axis kernel
    prewittX = numpy.array((
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]), dtype="int")

    img2 = cv2.filter2D(img_cv, -1, prewittX)

    cv2.namedWindow(winname="Prewitt X", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Prewitt X", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def prewitt_y():
    # construct the Prewitt y-axis kernel
    prewittY = numpy.array((
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]), dtype="int")

    img2 = cv2.filter2D(img_cv, -1, prewittY)

    cv2.namedWindow(winname="Prewitt Y", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Prewitt Y", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def roberts_corss():
    roberts_cross_v = numpy.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, -1]])

    roberts_cross_h = numpy.array([[0, 0, 0],
                                   [0, 0, 1],
                                   [0, -1, 0]])

    horizontal = cv2.filter2D(img_cv, -1, roberts_cross_v)
    vertical = cv2.filter2D(horizontal, -1, roberts_cross_h)

    img2 = img_cv.copy()
    h = img_cv.shape[0]
    w = img_cv.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            img2[y, x] = math.sqrt(vertical[y, x] ** 2 + horizontal[y, x] ** 2)

    cv2.namedWindow(winname="Roberts cross", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Roberts cross", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def zad_3_win():
    win = tkinter.Toplevel()
    win.wm_title("Zad3")

    b1 = ttk.Button(win, text="Sobel X", command=sobel_x)
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="Sobel Y", command=sobel_y)
    b2.grid(row=0, column=1, pady=20, padx=30)

    b3 = ttk.Button(win, text="Prewitt X", command=prewitt_x)
    b3.grid(row=1, column=0, pady=20, padx=30)
    b4 = ttk.Button(win, text="Prewitt Y", command=prewitt_y)
    b4.grid(row=1, column=1, pady=20, padx=30)

    b5 = ttk.Button(win, text="Robertson", command=roberts_corss)
    b5.grid(row=2, column=0, pady=20, padx=30)


def szkielet():
    print("hello")
    remain = True
    while (remain):
        remain = false


def erosion(tab):
    img2 = cv2.erode(img_cv, tab, iterations=1)

    cv2.namedWindow(winname="Erosion", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Erosion", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def dilate(tab):
    img2 = cv2.dilate(img_cv, tab, iterations=1)

    cv2.namedWindow(winname="Dilate", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Dilate", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def erosion_win():
    # Taking a matrix of size 5 as the kernel
    win = tkinter.Toplevel()
    win.wm_title("Erosion")

    b1 = ttk.Button(win, text="[0, 1, 0]\n[1, 1, 1]\n[0, 1, 0]", command=lambda: erosion(
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[1, 1, 1]\n[1, 1, 1]\n[1, 1, 1]", command=lambda: erosion(
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ))
    b2.grid(row=0, column=1, pady=20, padx=30)


def dilate_win():
    # Taking a matrix of size 5 as the kernel
    win = tkinter.Toplevel()
    win.wm_title("Dilate")

    b1 = ttk.Button(win, text="[0, 1, 0]\n[1, 1, 1]\n[0, 1, 0]", command=lambda: dilate(
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[1, 1, 1]\n[1, 1, 1]\n[1, 1, 1]", command=lambda: dilate(
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ))
    b2.grid(row=0, column=1, pady=20, padx=30)


def openn(tab):
    img2 = cv2.erode(img_cv, tab, iterations=1)
    img2 = cv2.dilate(img2, tab, iterations=1)

    cv2.namedWindow(winname="Open", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Open", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def closee(tab):
    img2 = cv2.dilate(img_cv, tab, iterations=1)
    img2 = cv2.erode(img2, tab, iterations=1)

    cv2.namedWindow(winname="Close", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="Close", mat=img2)
    plt.hist(img_cv.ravel(), bins=256, range=(0.0, 256.0), alpha=0.25, color='r')
    plt.hist(img2.ravel(), bins=256, range=(0.0, 256.0), color='g')
    plt.show()


def open_win():
    # Taking a matrix of size 5 as the kernel
    win = tkinter.Toplevel()
    win.wm_title("Open")

    b1 = ttk.Button(win, text="[0, 1, 0]\n[1, 1, 1]\n[0, 1, 0]", command=lambda: openn(
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[1, 1, 1]\n[1, 1, 1]\n[1, 1, 1]", command=lambda: openn(
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ))
    b2.grid(row=0, column=1, pady=20, padx=30)


def close_win():
    # Taking a matrix of size 5 as the kernel
    win = tkinter.Toplevel()
    win.wm_title("Close")

    b1 = ttk.Button(win, text="[0, 1, 0]\n[1, 1, 1]\n[0, 1, 0]", command=lambda: closee(
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ))
    b1.grid(row=0, column=0, pady=20, padx=30)
    b2 = ttk.Button(win, text="[1, 1, 1]\n[1, 1, 1]\n[1, 1, 1]", command=lambda: closee(
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ))
    b2.grid(row=0, column=1, pady=20, padx=30)


def distanceTransformL1():
    img = img_cv
    assert not isinstance(img, type(None)), 'image not found'
    # cv2.imshow('Source Image', img)
    # zmieniamy tlo na ciemne
    img[np.all(img == 255, axis=2)] = 0
    cv2.namedWindow(winname="Black Background", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Black Background', img)
    ## [sharp]
    # Create a kernel that we will use to sharpen our image
    # an approximation of second derivative, a quite strong kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated

    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    sharp = np.float32(img)
    imgResult = sharp - imgLaplacian

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    cv2.namedWindow(winname="New Sharped Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('New Sharped Image', imgResult)

    ## [bin]
    # Create binary image from source image
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.namedWindow(winname="Binary Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image', bw)
    ## [bin]

    ## [dist]
    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(bw, cv2.DIST_L1, 3)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.namedWindow(winname="Distance Transform Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Distance Transform Image', dist)
    ## [dist]


def distanceTransformC():
    img = img_cv
    assert not isinstance(img, type(None)), 'image not found'
    # cv2.imshow('Source Image', img)
    # zmieniamy tlo na ciemne
    img[np.all(img == 255, axis=2)] = 0
    cv2.namedWindow(winname="Black Background", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Black Background', img)
    ## [sharp]
    # Create a kernel that we will use to sharpen our image
    # an approximation of second derivative, a quite strong kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated

    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    sharp = np.float32(img)
    imgResult = sharp - imgLaplacian

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    cv2.namedWindow(winname="New Sharped Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('New Sharped Image', imgResult)

    ## [bin]
    # Create binary image from source image
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.namedWindow(winname="Binary Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image', bw)
    ## [bin]

    ## [dist]
    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(bw, cv2.DIST_C, 3)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.namedWindow(winname="Distance Transform Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Distance Transform Image', dist)
    ## [dist]


def euklides_mask():
    win = tkinter.Toplevel()
    win.wm_title("Wybor Mediany")

    l = tkinter.Scale(win, from_=3, to=5, tickinterval=2, orient=tkinter.HORIZONTAL, label="Rozmiar Maski")
    l.grid(row=0, column=0)

    b = ttk.Button(win, text="Okay", command=lambda: distanceTransform(l.get()))
    b.grid(row=1, column=0, pady=20, padx=30)
    return


def distanceTransform(mask):
    img = img_cv
    assert not isinstance(img, type(None)), 'image not found'
    # cv2.imshow('Source Image', img)
    # zmieniamy tlo na ciemne
    img[np.all(img == 255, axis=2)] = 0
    cv2.namedWindow(winname="Black Background", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Black Background', img)
    ## [sharp]
    # Create a kernel that we will use to sharpen our image
    # an approximation of second derivative, a quite strong kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated

    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    sharp = np.float32(img)
    imgResult = sharp - imgLaplacian

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    cv2.namedWindow(winname="New Sharped Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('New Sharped Image', imgResult)

    ## [bin]
    # Create binary image from source image
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.namedWindow(winname="Binary Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image', bw)
    ## [bin]

    ## [dist]
    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, mask)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.namedWindow(winname="Distance Transform Image", flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Distance Transform Image', dist)
    ## [dist]


menubar = tkinter.Menu(root)

# create a pulldown menu, and add it to the menu bar
filemenu = tkinter.Menu(menubar, tearoff=0)
filemenu.add_command(label="Monochromatic", command=load_mono)
filemenu.add_command(label="Color", command=load_color)
menubar.add_cascade(label="Obraz", menu=filemenu)

# create more pulldown menus
lab1_menu = tkinter.Menu(menubar, tearoff=0)
lab1_menu.add_command(label="Histogram", command=histogram)
menubar.add_cascade(label="Laboratorium 1", menu=lab1_menu)

# lab2 menu
lab2_menu = tkinter.Menu(menubar, tearoff=0)
lab2_menu.add_command(label="Streching", command=rozciaganie)
lab2_menu.add_command(label="Equalization", command=equalizacja)
lab2_menu.add_separator()
lab2_menu.add_command(label="Negation", command=negacja)
lab2_menu.add_command(label="Threshold", command=progowanie)
lab2_menu.add_command(label="Threshold with original", command=progowanie_z_zachowaniem)
lab2_menu.add_command(label="Grayscale reduction", comman=poziomy_szar)
lab2_menu.add_command(label="Streching paramed", command=rozciaganie_param)
menubar.add_cascade(label="Laboratorium 2", menu=lab2_menu)

# lab3 menu
lab3_menu = tkinter.Menu(menubar, tearoff=0)
lab3_menu.add_command(label="Mediana", command=mediana_win)
lab3_menu.add_command(label="Smooting default", command=smoothing_win)
lab3_menu.add_command(label="Laplacian default", command=laplacian_win)
lab3_menu.add_command(label="Edge detection", command=detection_win)
lab3_menu.add_command(label="Zad 3", command=zad_3_win)
menubar.add_cascade(label="Laboratorium 3", menu=lab3_menu)

# lab4 menu
lab4_menu = tkinter.Menu(menubar, tearoff=0)
lab4_menu.add_command(label="Erosion", command=erosion_win)
lab4_menu.add_command(label="Dilate", command=dilate_win)
lab4_menu.add_command(label="Open", command=open_win)
lab4_menu.add_command(label="Close", command=close_win)
menubar.add_cascade(label="Laboratorium 4", menu=lab4_menu)

# lab5 menu

lab5_menu = tkinter.Menu(menubar, tearoff=0)
lab5_menu.add_command(label="Segmentation")  # command=segmentation
lab5_menu.add_command(label="Shape Features")
lab5_menu.add_command(label="Moment Descriptors")
lab5_menu.add_command(label="Central Moment")
menubar.add_cascade(label="Laboratorium 5", menu=lab5_menu)

# menu
lab6_menu = tkinter.Menu(menubar, tearoff=0)
lab6_menu.add_command(label="O mnie")
lab6_menu.add_command(label="Info")
menubar.add_cascade(label="O Programie", menu=lab6_menu)

# projekt

projekt_menu = tkinter.Menu(menubar, tearoff=0)
projekt_menu.add_command(label="Transformata Euklidesowa", command=euklides_mask)
projekt_menu.add_command(label="Transformata L1", command=distanceTransformL1)
projekt_menu.add_command(label="Transformata C", command=distanceTransformC)
menubar.add_cascade(label="Projekt", menu=projekt_menu)

menubar.add_cascade(label="Zamknij", command=exit)


def exit(self):
    self.frame.destroy()


# display the menu
root.config(menu=menubar)
tkinter.Label(root, text="                                                                              ",
              background='white').pack()
tkinter.Label(root, text="Jaroslaw Gronowski WIT IZ07IO1", background='white').pack()
tkinter.Label(root, text="       Zaladuj obraz aby moc wykonywac operacje           ", background='white').pack()
tkinter.Label(root, text="      W przypadku transformaty odleglosciowej - obraz powinien byc kolorowy          ",
              background='white').pack()
tkinter.Label(root, text="                                                                              ",
              background='white').pack()
tkinter.Label(root, text="                                                                              ",
              background='white').pack()

root.mainloop()