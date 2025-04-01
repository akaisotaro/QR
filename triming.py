from PIL import Image

# 画像を開く
image = Image.open("1.png")

# 中央部分をトリミング
left = (image.width - 128) // 2  # 横方向の開始位置
top = 0  # 縦方向はそのまま
right = left + 128  # 横方向の終了位置
bottom = 128  # 縦方向の終了位置

cropped_image = image.crop((left, top, right, bottom))

# 画像を保存
cropped_image.save("2.png")

print("中央部分をトリミングして2.pngとして保存しました")
