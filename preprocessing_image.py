import os
import numpy as np
import cv2


def preprocessing_images(input_path, output_path):
    try:
        os.mkdir(output_path)
    except:
        pass
    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, filename))
        if img is not None:
            final_img = img.copy()
            if len(final_img.shape) == 2:
                final_img = final_img[:, :, np.newaxis]
                final_img = np.concatenate([final_img, final_img, final_img], axis=2)
            final_img = cv2.resize(final_img, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_path, filename), final_img)

