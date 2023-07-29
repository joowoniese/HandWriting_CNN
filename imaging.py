from PIL import Image, ImageDraw, ImageFont

image_width = 28
image_height = 28

#폰트 크기랑 글씨 위치가 정가운데인지 한 번만 확인해주세요
for i in range(10):
    image = Image.new("L", (image_width, image_height), color=0)
    draw = ImageDraw.Draw(image)

    #size가 폰트 크기입니다
    font = ImageFont.truetype("./font/헬스셋복조리Std.ttf", size=25)  #font디렉토리 안에있는 '강원교육튼튼.ttf'를 폰트로 설정
                                                                 #'강원교육튼튼.ttf'부분만 바꿔주시면 돼요.
    number = str(i)
    text_bbox = draw.textbbox((0, 0), number, font=font)
    x = (image_width - text_bbox[2]) // 2

    #y_offset을 늘리고 줄여가면서 글자 위치를 확인해주세요
    #숫자가 클수록 아래로 내려가고, 작을수록 위로 올라갑니다(음수도 가능)
    y_offset = -4

    y = (image_height - text_bbox[3]) // 2 + y_offset
    draw.text((x, y), number, fill=255, font=font)

    #한글자당 3개씩
    for j in range(3):
        #teun부분만 파일명 바꿔주시면 돼요. 반드시 영어로 이름 지어주세요.
        image.save("./handwritingdata/" + str(i) + "helathset.png")