import cv2
import numpy as np
cap = cv2.VideoCapture('12233.mp4')

cap = cv2.VideoCapture(0)
#cap.set(3, 1280)  # CV_CAP_PROP_FRAME_WIDTH
#cap.set(4, 720)  # CV_CAP_PROP_FRAME_HEIGHT
#cap.set(5, 0)  # CV_CAP_PROP_FPS


def region_of_interest(img, vertices) :
    mask = np.zeros_like(img)

    ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a, b, c):
    return cv2.addWeighted(initial_img, a, img, b, c)


rho = 2
theta = np.pi / 180
threshold = 90
min_line_len = 30
max_line_gap = 250

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # print(frame.shape)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_frame = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 250 #이거 작을수록 더 자세하게 되는듯
    edges = cv2.Canny(blur_frame, low_threshold, high_threshold)
    mask = np.zeros_like(edges)

    ignore_mask_color = 255

    vertices = np.array([[(0, 1500),
                          (30, 0),
                          (1500, 0),
                          (1580, 1500)]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    mask = region_of_interest(edges, vertices)
    lines = hough_lines(mask, rho, theta, threshold,
                        min_line_len, max_line_gap)

    lines_edges = weighted_img(lines, frame, a=0.8, b=1.0, c=0.0)

    cv2.imshow('Lane Detection', lines_edges)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 2,
                "id": "9679b761",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import cv2\n",
                    "import numpy as np"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "id": "d025a84d",
                "metadata": {},
                "outputs": [],
                "source": [
                    "def draw_lines(img, lines, color=[255,0,0],thickness=5):\n",
                    "    for line in lines:\n",
                    "        if line is None:\n",
                    "            print('none')\n",
                    "        else:\n",
                    "            for x1, y1, x2, y2 in line:\n",
                    "                cv2.line(img, (x1,y1), (x2,y2), color, thickness)\n",
                    "            \n",
                    "def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):\n",
                    "    lines=cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),\n",
                    "                         minLineLength=min_line_len,\n",
                    "                         maxLineGap=max_line_gap)\n",
                    "    line_img = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)\n",
                    "    draw_lines(line_img, lines)\n",
                    "    return line_img\n",
                    "\n",
                    "def weighted_img(img, initial_img, a=0.75, b=1.0, c=0.0):\n",
                    "    return cv2.addWeighted(initial_img, a, img, b, c)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 17,
                "id": "a9ddcfda",
                "metadata": {},
                "outputs": [],
                "source": [
                    "\n",
                    "width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)\n",
                    "height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)\n",
                    "\n",
                    "vertices=np.array([[(300, 720), #왼쪽 아래\n",
                    "                   (600,450),     #왼쪽 위\n",
                    "                   #(700,400),    #오른쪽 위\n",
                    "                   (900, 720)]], dtype=np.int32)   #오른쪽 아래\n",
                    "ignore_mask_color = (255,)*3\n",
                    "\n",
                    "kernel_size = 5\n",
                    "\n",
                    "low_threshold = 50\n",
                    "high_threshold = 150\n",
                    "\n",
                    "rho = 2\n",
                    "theta = np.pi/180\n",
                    "threshold=90\n",
                    "min_line_len = 100\n",
                    "max_line_gap = 150"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "f4eac5cf",
                "metadata": {},
                "outputs": [],
                "source": [
                    "while cap.isOpened() :\n",
                    "    ret, frame = cap.read()\n",
                    "    if not ret :\n",
                    "        break\n",
                    "    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
                    "    blur_frame = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)\n",
                    "    edges = cv2.Canny(blur_frame, low_threshold, high_threshold)\n",
                    "    mask = np.zeros_like(edges)\n",
                    "    mask_ = cv2.fillPoly(mask, vertices, ignore_mask_color)\n",
                    "    masked_image = cv2.bitwise_and(edges, mask_)\n",
                    "    lines = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)\n",
                    "    lines_edges=weighted_img(lines, frame, a=0.8, b=1.0, c=0.0)\n",
                    "    \n",
                    "    cv2.imshow('lines_edges',lines_edges)\n",
                    "    cv2.waitKey(1)\n",
                    "\n",
                    "cap.release()\n",
                    "cv2.destroyAllWindows()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "5de7d432",
                "metadata": {},
                "outputs": [],
                "source": []
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "e2d0061d",
                "metadata": {},
                "outputs": [],
                "source": []
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

cap.release()
cv2.destroyAllWindows()