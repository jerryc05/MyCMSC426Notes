from PIL import Image, ImageDraw

im = Image.new('1', (160, 160))

draw = ImageDraw.Draw(im)
draw.polygon([20, 20, 20, 140, 60, 60], fill='white')

im.save('triangle1.png')