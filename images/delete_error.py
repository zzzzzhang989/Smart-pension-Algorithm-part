import os
import numpy
import imghdr
from PIL import Image

# 删除不是JPEG或者PNG格式的图片
def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf == b'\xff\xd9'  # 判定jpg是否包含结束字段


def delete_error_image(image_dir):
    try:
        images = os.listdir(image_dir)
        for image in images:
            image = os.path.join(image_dir, image)
            try:
                # 获取图片的类型
                image_type = imghdr.what(image)
                # 如果图片格式不是jpg或者jpg格式有误则删除
                ret = is_valid_jpg(image)
                if image_type != 'jpeg' or not ret:
                    os.remove(image)
                    print('已删除：%s' % image)
                    continue
                # 删除灰度图
                img = numpy.array(Image.open(image))
                if len(img.shape) == 2:
                    os.remove(image)
                    print('已删除：%s' % image)
            except:
                os.remove(image)
                print('已删除：%s' % image)
    except:
        pass


if __name__ == '__main__':
    delete_error_image('./faces/98')
