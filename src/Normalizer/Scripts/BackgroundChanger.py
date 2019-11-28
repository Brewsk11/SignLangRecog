from image_fetcher.multithread_image_fetching import concurrent_image_search, concurrent_images_download
from image_fetcher.browsers import Browser, BrowserType
from PIL import Image
from PIL.ImageColor import getrgb
from random import randint
from os import listdir, mkdir
from os.path import isdir, isfile

from Normalizer.Providers.DirectoryImageProvider import  DirectoryImageProvider as ImgProvider
from Normalizer.Models.ImageModels import TrainingImage, TaggedImage


def cutout_black(img: Image.Image):
    imgdata = img.load()

    w, h = img.size
    for y in range(h):
        for x in range(w):
            if imgdata[x, y] == (0, 0, 0, 255):
                imgdata[x, y] = (0, 0, 0, 0)

    return img


def crop_rand_fragment(img: Image.Image, size: tuple):
    width, height = img.size
    if width < size[0] or height < size[1]:
        raise ValueError(f'Image provided has a size of {img.size} but you want to cut out a frogment of size {size}')

    from_x = randint(0, width - size[0])
    from_y = randint(0, height - size[1])

    return img.crop((from_x, from_y, from_x + size[0], from_y + size[1]))


def zoom(img: Image.Image, pixels: int):
    width, height = img.size
    img = img.resize((width + pixels * 2, height + pixels * 2), resample=Image.BICUBIC)
    img = img.crop((pixels, pixels, width + pixels, height + pixels))
    return img


def augment(img_org: Image.Image, img_tgg: Image.Image, zoom_range: tuple, rotation_range: tuple):
    rotation_angle: int = randint(rotation_range[0], rotation_range[1])
    zoom_by: int = randint(zoom_range[0], zoom_range[1])

    img_org = img_org.rotate(rotation_angle, resample=Image.BICUBIC)
    img_org = zoom(img_org, zoom_by)
    img_tgg = img_tgg.rotate(rotation_angle, resample=Image.BICUBIC)
    img_tgg = zoom(img_tgg, zoom_by)
    return img_org, img_tgg


def generate_new_image(bg: Image.Image, org: Image.Image, tgg: Image.Image):
    cropped_bg = crop_rand_fragment(bg, org.size)
    tgg_alpha = cutout_black(tgg)
    aug_org, aug_tgg_alpha = augment(org, tgg_alpha, (10, 20), (-45, 45))

    aug_tgg = Image.new('RGBA', aug_tgg_alpha.size, getrgb('black'))
    aug_tgg = Image.alpha_composite(aug_tgg, aug_tgg_alpha)

    gen = Image.new('RGBA', aug_org.size)
    gen.paste(cropped_bg, (0, 0), cropped_bg)
    # gen.paste(aug_org, (0, 0), aug_tgg_alpha)

    return gen, aug_tgg


if __name__ == "__main__":
    img_path = '/home/pawel/PracaInzynierska/tmp'
    background_paths = ['/home/pawel/PracaInzynierska/room_bg']
    org_path = '/home/pawel/PracaInzynierska/TrainingData'
    tgg_path = '/home/pawel/PracaInzynierska/LiterkiTagged'

    new_org_path = '/home/pawel/PracaInzynierska/TrainingData_NewerBG_nothing'
    new_tgg_path = '/home/pawel/PracaInzynierska/LiterkiTagged_NewerBG_nothing'

    bg_imgs = []
    for pth in background_paths:
        img_names = listdir(pth)
        for img_name in img_names:
            bg_imgs.append(Image.open(pth + '/' + img_name))

    tgg_imgs = ImgProvider(tgg_path, TaggedImage)
    org_imgs_all = ImgProvider(org_path, TrainingImage)
    tgg_imgs = tgg_imgs.resolution(128)

    org_imgs = []
    for img in org_imgs_all.all:
        for tgg_img in tgg_imgs:
            if tgg_img.number == img.number and tgg_img.letter == img.letter:
                org_imgs.append(img)
                break

    org_imgs.sort(key=lambda x: (x.letter, x.number))
    tgg_imgs.sort(key=lambda x: (x.letter, x.number))

    for i in range(len(org_imgs)):

        org = org_imgs[i].pillow_image.resize((128, 128), resample=Image.BICUBIC).convert('RGBA')
        tgg = tgg_imgs[i].pillow_image.convert('RGBA')

        for j in range(1):

            bg_w, bg_h = 0, 0
            while bg_w < 128 or bg_h < 128:
                bg_img_num = randint(0, len(bg_imgs)-1)
                bg_img = bg_imgs[bg_img_num].convert('RGBA')
                bg_w, bg_h = bg_img.size

            new_org, new_tgg = generate_new_image(bg_img, org, tgg)

            if not isdir(new_org_path + '/' + org_imgs[i].letter):
                mkdir(new_org_path + '/' + org_imgs[i].letter)

            # if not isdir(new_tgg_path + '/' + tgg_imgs[i].letter):
                # mkdir(new_tgg_path + '/' + tgg_imgs[i].letter)

            new_org.convert('RGB').resize((128, 128), resample=Image.BICUBIC).save(new_org_path + '/' + org_imgs[i].letter + '/' + org_imgs[i].filename.split('.')[0] + f'{j:02d}.jpg')
            # new_tgg.convert('RGB').resize((128, 128), resample=Image.BICUBIC).save(new_tgg_path + '/' + tgg_imgs[i].letter + '/' + tgg_imgs[i].filename.split('_')[0] + f'{j:02d}' + '_128.bmp')

        if i % 50 == 0:
            print(f'[{i}/{len(org_imgs)}]')

    """concurrent_images_download(
        search_term='room photo',
        max_image_fetching_threads=20,
        image_download_timeout=5,
        total_images=200,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 '
                               'Safari/537.36'},
        browser=Browser(BrowserType.FIREFOX, '/home/pawel/geckodriver'),
        extensions=['jpg']
    )"""

    """tgg = Image.open(img_path + '/tgg_128.bmp').convert('RGBA')
    tgg = tgg.resize((200, 200), resample=Image.BICUBIC).convert('RGBA')
    org = Image.open(img_path + '/org_200.jpg').convert('RGBA')
    bg = Image.open(img_path + '/bg.jpg').convert('RGBA')

    new_org, new_tgg = generate_new_image(bg, org, tgg)
    new_org.show()
    new_tgg.show()"""
