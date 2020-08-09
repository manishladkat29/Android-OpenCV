import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

srcDir = os.path.join('./images', 'template_matching')
srcTemplateDir = os.path.join('./images', 'template_matching', 'template')
dstDir = os.path.join('./images', 'template_matching', 'results')
srcDirContent = os.listdir(srcDir)
for input in srcDirContent:
    if os.path.isfile(os.path.join(srcDir, input)):
        fileName = os.path.splitext(os.path.basename(input))[0]
        templateImageFileName = fileName + '-template.jpg'
        img = cv2.imread(os.path.join(srcDir, input), 0)
        img2 = img.copy()
        template = cv2.imread(os.path.join(srcTemplateDir, templateImageFileName), 0)
        w, h = template.shape[::-1]
        # 'cv2.TM_CCOEFF' removed since it was a bit off
        methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            res = cv2.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, (top_left[1]) + h)

            cv2.rectangle(img, top_left, bottom_right, 255, 2)
            # cv2.imwrite(os.path.join(dstDir, fileName + meth + ".jpg"), img)

            contours, hierarchy = cv2.findContours(img, 1, 2)
            cnt = contours[0]
            M = cv2.moments(cnt)
            print(M)
            x, y, w, h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


            # image[y:y + h, x:x + w]
            # Select ROI
            # r = cv2.selectROI(img)
            # # Crop image
            # r = cv2.selectROI(img)
            # imCrop = img[top_left, bottom_right]
            cv2.imwrite(os.path.join(dstDir, fileName + meth + "_cropped.jpg"), img[y:y + h, x:x + w])
    print("--------------")
#
# img = cv2.imread(os.path.join("./images", "template_matching", "sy-2.jpg"),0)
# img2 = img.copy()
# template = cv2.imread(os.path.join("./images", "template_matching", "sy-2-template.jpg"),0)
# w, h = template.shape[::-1]
# print(w,h)
#
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#
#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, (top_left[1]+100) + h-350)
#
#     cv2.rectangle(img,top_left, bottom_right, 255, 2)
#
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#
#     plt.show()


# im = cv2.imread('c:/data/ph.jpg')
# gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
