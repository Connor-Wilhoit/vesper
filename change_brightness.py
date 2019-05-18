def change_brightness(image, brightness_factor):
    hsv_image          = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_factor
    image_rgb          = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb
