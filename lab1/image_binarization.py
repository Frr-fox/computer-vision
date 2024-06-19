import cv2
import numpy as np
from numba import jit

# с использованием OpenCV
def adaptive_binarization_cv2(image, block_size=21, C=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    return binary_image

# с использованием OpenCV
def to_black_white_cv2(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# без использования OpenCV
def to_black_white_py(image):
    return np.dot(image[...,:3], [0.2989, 0.587, 0.114]).astype(np.uint8)

# без использования OpenCV
def adaptive_binarization_py(image, block_size=21, C=10):
    gray_image = to_black_white_py(image)
    height, width = gray_image.shape
    binary_image = np.zeros((height, width), dtype=np.uint8)
    half_block = block_size // 2

    for i in range(height):
        for j in range(width):
            y1 = max(0, i - half_block)
            y2 = min(height, i + half_block + 1)
            x1 = max(0, j - half_block)
            x2 = min(width, j + half_block + 1)
            
            block = gray_image[y1:y2, x1:x2]
            mean = np.mean(block)
            
            if gray_image[i, j] > mean - C:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image

# с использованием Numba
@jit
def to_black_white_jit(image):
    x, y, _ = image.shape
    arr = np.zeros((x, y), dtype=np.uint8)
    
    for i in range(x):
        for j in range(y):
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            gray = 0.2989 * r + 0.587 * g + 0.114 * b
            arr[i, j] = int(gray)
    
    return arr

# с использованием Numba
@jit
def adaptive_binarization_jit(image, block_size=20, C=10):
    height, width = image.shape
    
    binary_image = np.zeros((height, width), dtype='uint8')
    
    half_block = block_size // 2
    
    
    for i in range(height):
        for j in range(width):
            y1 = max(0, i - half_block)
            y2 = min(height, i + half_block + 1)
            x1 = max(0, j - half_block)
            x2 = min(width, j + half_block + 1)
            
            block = image[y1:y2, x1:x2]
            
            mean = np.mean(block)
            
            if image[i, j] > mean - C:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0
    
    return binary_image

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    mode = 'cv2_binary'

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Не удалось захватить изображение")
            break

        if mode == 'cv2_binary':
            output_image = adaptive_binarization_cv2(frame)
        elif mode == 'cv2_gray':
            output_image = to_black_white_cv2(frame)
        elif mode == 'py_binary':
            output_image = adaptive_binarization_py(frame)
        elif mode == 'py_gray':
            output_image = to_black_white_py(frame)
        elif mode == 'jit_binary':
            gray_image = to_black_white_jit(frame)
            output_image = adaptive_binarization_jit(gray_image, 21, 10)
        elif mode == 'jit_gray':
            output_image = to_black_white_jit(frame)

        cv2.imshow('Output', output_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if 'binary' in mode:
                mode = mode.replace('binary', 'gray')
            else:
                mode = mode.replace('gray', 'binary')
        elif key == ord('1'):
            if 'cv2' not in mode:
                mode = 'cv2_' + mode.split('_')[1]
        elif key == ord('2'):
            if 'py' not in mode:
                mode = 'py_' + mode.split('_')[1]
        elif key == ord('3'):
            if 'jit' not in mode:
                mode = 'jit_' + mode.split('_')[1]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
