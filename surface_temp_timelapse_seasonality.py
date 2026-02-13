import ee
import geemap
import os
import subprocess
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# --- START TELEMETRY ---
start_time = datetime.now()
print(f"🚀 Seasonal Analysis Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Initialize
project_id = 'surface-temp-trend' 
try:
    ee.Initialize(project=project_id)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=project_id)

roi = ee.Geometry.Rectangle([-85.35, 35.00, -85.15, 35.18])
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'Chatt_Pro_Pulse')
if not os.path.exists(out_dir): os.makedirs(out_dir)

ffmpeg_path = r"C:\Program Files\ShareX\ffmpeg.exe" 
font_path = r"C:\Windows\Fonts\arialbd.ttf"

# 2. Guided Thermal Function (Thermal + Roads)
def get_guided_lst(year, month):
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    def apply_scale(img):
        return img.select(['ST_B10'], ['LST']).multiply(0.00341802).add(149.0).subtract(273.15)
    
    fallback = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(roi)
                .filter(ee.Filter.calendarRange(month, month, 'month'))
                .filter(ee.Filter.calendarRange(2014, 2024, 'year'))
                .map(apply_scale).median().clip(roi))

    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(roi).filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUD_COVER', 45)))

    lst_img = ee.Image(col.map(apply_scale).median()).clip(roi).unmask(fallback) if col.size().getInfo() > 0 else fallback
    
    # Guidance Layer: TIGER Roads
    highways = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(roi)
    highway_overlay = ee.Image().paint(highways, 1, 1).visualize(palette=['white'], opacity=0.5)
    
    vis_thermal = lst_img.visualize(min=0, max=40, palette=['blue', 'cyan', 'green', 'yellow', 'orange', 'red'])
    return vis_thermal.blend(highway_overlay)

# 3. Processing Loop (Seasonal Sequence + Cut Scenes)
years = range(2014, 2026)
months = range(1, 13)
month_names = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
short_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

seasonal_sequence = []

for mo in months:
    # --- CREATE CUT SCENE FRAME ---
    cut_path = os.path.join(out_dir, f"cutscene_{mo:02d}.png")
    cut_img = Image.new('RGB', (768, 768), color=(0, 0, 0))
    draw = ImageDraw.Draw(cut_img)
    try:
        title_font = ImageFont.truetype(font_path, 60)
        sub_font = ImageFont.truetype(font_path, 40)
    except:
        title_font = sub_font = ImageFont.load_default()
    
    draw.text((384, 340), f"{month_names[mo-1]}", fill="white", font=title_font, anchor="mm")
    draw.text((384, 400), "Year over Year Trend", fill="white", font=sub_font, anchor="mm")
    cut_img.save(cut_path)
    
    # Title card lasts 2.0s for a professional pause
    seasonal_sequence.append({'path': cut_path, 'duration': 2.0})

    for yr in years:
        # Note: We use a specific filename for guided frames to avoid conflict with old ones
        png_path = os.path.join(out_dir, f"guided_{mo:02d}_{yr}.png")
        if not os.path.exists(png_path) or os.path.getsize(png_path) < 50000:
            print(f"🛰️ Processing Season: {short_names[mo-1]} {yr}...")
            img = get_guided_lst(yr, mo)
            geemap.get_image_thumbnail(img, png_path, {}, dimensions=768, region=roi)
        
        # Apply Shrunk 55pt Label
        try:
            label_img = Image.open(png_path).convert('RGB')
            draw = ImageDraw.Draw(label_img)
            date_text = f"{short_names[mo-1]} {yr}"
            font = ImageFont.truetype(font_path, 55)
            draw.rectangle([10, 10, 310, 85], fill=(0, 0, 0))
            draw.text((25, 18), date_text, fill="white", font=font)
            label_img.save(png_path)
        except Exception as e: print(f"⚠️ Label error: {e}")
        
        seasonal_sequence.append({'path': png_path, 'duration': 1.0})

# 4. Final Video Render (RTX 4070 Optimized)
output_mp4 = os.path.join(out_dir, "hixson_seasonal_analysis.mp4")
file_list_path = os.path.join(out_dir, "ffmpeg_input.txt")

print("\n🎬 Blending Seasonal Trends using RTX 4070...")

with open(file_list_path, 'w') as f:
    for item in seasonal_sequence:
        f.write(f"file '{os.path.basename(item['path'])}'\n")
        f.write(f"duration {item['duration']}\n")
    if seasonal_sequence:
        f.write(f"file '{os.path.basename(seasonal_sequence[-1]['path'])}'\n")

ffmpeg_cmd = [
    ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', file_list_path,
    '-vf', 'minterpolate=fps=60:mi_mode=blend,format=yuv420p', 
    '-c:v', 'h264_nvenc', '-preset', 'p7', '-rc', 'vbr', '-cq', '20', '-fps_mode', 'vfr',
    output_mp4
]

try:
    if os.path.exists(output_mp4): os.remove(output_mp4)
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"🔥 ANALYSIS VIDEO COMPLETE: {output_mp4}")
except Exception as e: print(f"❌ FFmpeg error: {e}")

# --- END TELEMETRY ---
end_time = datetime.now()
print(f"⏱️ Total Runtime: {end_time - start_time}")