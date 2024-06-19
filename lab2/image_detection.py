import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def template_matching_cv(image_path, template_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    methods = [
        # 'cv2.TM_CCOEFF', 
        'cv2.TM_CCOEFF_NORMED', 
        # 'cv2.TM_CCORR', 
        # 'cv2.TM_CCORR_NORMED', 
        # 'cv2.TM_SQDIFF', 
        # 'cv2.TM_SQDIFF_NORMED'
               ]
    
    for method in methods:
        meth = eval(method)
        image_copy = image.copy()
        result = cv2.matchTemplate(image_gray, template_gray, meth)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
         # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        h, w = template_gray.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)
        
        plt.subplot(1, 2, 1)
        plt.imshow(template)
        plt.title('Template')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title('Template Matching Result with ' + method[4:])
        plt.axis('off')
        plt.show()

    return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]

def iteration(template_gray, image_gray):
    h, w = template_gray.shape
    min_ssd = float('inf')
    top_left = (0, 0)

 # Перебор всех возможных положений шаблона в изображении
    for y in range(image_gray.shape[0] - h + 1):
        for x in range(image_gray.shape[1] - w + 1):
            roi = image_gray[y:y+h, x:x+w]
            ssd = np.sum((roi - template_gray)**2)
            if ssd < min_ssd:
                min_ssd = ssd
                top_left = (x, y)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right

def template_matching(image_path, template_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    top_left, bottom_right = iteration(template_gray, image_gray)
    
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(template)
    plt.title('Template')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title('Direct Template Matching Result')
    plt.axis('off')
    plt.show()

    return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]

def keypoint_matching(image_path, template_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(image_gray, None)

    if des1 is None or des2 is None:
        print("Ключевые точки не нашлись на изображениях")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # draw first 50 matches
    img3 = cv2.drawMatches(template_gray,kp1,image_gray,kp2,matches[:50], None, flags=2)
    plt.imshow(img3),plt.show()

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = template_gray.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    top_right = (np.int32(dst)[0][0][0], np.int32(dst)[0][0][1])
    top_left = (np.int32(dst)[1][0][0], np.int32(dst)[1][0][1])
    bottom_left = (np.int32(dst)[2][0][0], np.int32(dst)[2][0][1])
    bottom_right = (np.int32(dst)[3][0][0], np.int32(dst)[3][0][1])

    image = cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(template)
    plt.title('Template')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title('Keypoint Matching Result')
    plt.axis('off')
    plt.show()

    return [top_right, top_left, bottom_left, bottom_right]


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    iou = interArea / unionArea

    return iou

def calculate_rotated_iou(boxA, boxB):
    # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    polyA = Polygon(boxA)
    polyB = Polygon(boxB)
    try:
        interArea = polyA.intersection(polyB).area
        unionArea = polyA.area + polyB.area - interArea
        iou = interArea / unionArea if unionArea != 0 else 0
    except:
        iou = 0
    return iou

def convert_to_4_points(bbox):
    height = bbox[3] - bbox[1]     # 2    1
    width = bbox[2] - bbox[0]      # 3    4
    return [(bbox[2], bbox[1]), (bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])]


def get_annotations(annotation_path):
    ann_map = {}
    with open(annotation_path, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            id, x1, y1, x2, y2 = line.split(';')
            if id == 'id':
                continue
            ann_map[int(id)] = [int(x1), int(y1), int(x2), int(y2)]

    return ann_map