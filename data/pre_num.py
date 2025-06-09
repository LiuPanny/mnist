from PIL import Image, ImageDraw
import numpy as np
import random

# 设置保存路径
save_path = "pre/"

# 创建空白图像
width, height = 28, 28  # 设置图像的宽度和高度

# 随机生成 20 个手写数字图片并保存
for i in range(20):
    image = Image.new('L', (width, height), color=0)  # 'L' 表示灰度图像
    draw = ImageDraw.Draw(image)

    # 随机选择一个数字
    digit = random.randint(0, 9)

    # 随机生成绘制文本的位置
    x = random.randint(0, width - 8)  # x 坐标范围：0 到 width-8
    y = random.randint(0, height - 8)  # y 坐标范围：0 到 height-8

    # 在图像上绘制手写数字
    draw.text((x, y), str(digit), fill=255)  # 在图像的随机位置绘制数字

    # 保存图像
    image.save(save_path + f"digit_{i}_{digit}.png")

print("Images saved successfully.")