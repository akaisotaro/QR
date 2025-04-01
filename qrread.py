from pyzbar.pyzbar import decode
from PIL import Image

# QRコード画像を読み込み
qr_image = Image.open("qr_code.png")

# QRコードをデコード
decoded_objects = decode(qr_image)

# QRコードに含まれるデータ（URL）を出力
for obj in decoded_objects:
    if obj.type == 'QRCODE':
        print("QRコードのURL:", obj.data.decode('utf-8'))
