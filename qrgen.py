import qrcode
from PIL import Image, Resampling

# QRコードのデータ
data = "https://youtu.be/asLRL5r2f-E"

# QRコードを生成
qr = qrcode.QRCode(
    version=1,  # バージョン（サイズ調整用、1が最小）
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # 誤り訂正レベル（L: 約7%復元可能）
    box_size=10,  # 1つのセルのピクセル数（後でリサイズするので大きめに）
    border=1,  # 外枠のセル数（最小限にする）
)

qr.add_data(data)
qr.make(fit=True)

# 画像として生成
qr_image = qr.make_image(fill="black", back_color="white")

# 128×128ピクセルにリサイズ（Resampling.NEARESTを使用）
qr_image = qr_image.resize((128, 128), Resampling.NEAREST)

# 画像を保存
qr_image.save("qr_code.png")

print("128×128のQRコードを 'qr_code.png' に保存しました。")
