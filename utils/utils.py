def make_even(number):
    _, remainder = divmod(number, 2)
    return number + remainder


def image_posterize(image, threshold):
    image = image.mean(dim=3, keepdim=True)
    image = (image > threshold).float()
    image = image.repeat(1, 1, 1, 3)

    return image
