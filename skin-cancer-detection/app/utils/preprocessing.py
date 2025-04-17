def resize_image(image, height, width):
    return cv2.resize(image, (width, height))

def normalize_image(image, mean, std):
    return (image - mean) / std

def preprocess_image(image_path, height=256, width=256, mean=0, std=1):
    image = cv2.imread(image_path)
    image = resize_image(image, height, width)
    image = normalize_image(image, mean, std)
    return image