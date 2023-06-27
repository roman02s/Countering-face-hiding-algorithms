import matplotlib.pyplot as plt
from skimage import color


def show_image(image, target):
    """
	Функция выводит изображение из датасета LFW People.
	Parameters:
	image (array-like): массив с изображением.
	target (array-like): массив с меткой класса.
	"""
    image = color.gray2rgb(image.reshape(-1, 50, 37))
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(image[0], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(target)
    plt.show()
