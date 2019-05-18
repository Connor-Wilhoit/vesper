def preprocess_image(image):
    image_cropped = image[25:375, :]
    image = cv2.resize(image_cropped, (220, 66), interpolation=cv2.INTER_AREA)
    return image

