def preprocess_image_from_path(image_path, speed, brightness_factor):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, brightness_factor)
    img = preprocess_image(img)
    return img, speed

