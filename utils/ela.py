from PIL import Image, ImageChops, ImageEnhance
import io

def compute_ela(image_path, quality=90, scale=15):
    orig = Image.open(image_path).convert('RGB')
    
    buffer = io.BytesIO()
    orig.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    
    ela_img = ImageChops.difference(orig, resaved)
    
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_factor = 255.0 / max_diff
    
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale_factor)
    return ela_img