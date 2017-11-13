from train import train
from predict import CnnModel
from captcha.image import ImageCaptcha

if __name__ == "__main__":
    image = ImageCaptcha(width=100,fonts=['/root/Symbol.ttf'])
    image.write('2580', './data/2580.png')
    a = CnnModel()
    print a.Calculate('./data/2580.png')