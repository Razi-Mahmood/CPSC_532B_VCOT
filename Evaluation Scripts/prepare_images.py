"""Create numbered original/lightly blurred pilot pairs without changing source images."""
import argparse,csv,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageFilter

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,default=Path('data/processed/images'))
 p.add_argument('--out',type=Path,default=Path('Blurred Image Examples'))
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];seen=set()
 for path in sorted(a.source.glob('*')):
  if path.suffix.lower() not in ['.png','.jpg','.jpeg','.webp']:continue
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest in seen:continue
  seen.add(digest);i=len(rows)+1
  image=Image.open(path).convert('RGB');radius=.01*min(image.size)
  first=f'Image {i:02d}.png';second=f'Image {i:02d} blurred.png'
  image.save(a.out/first);image.filter(ImageFilter.GaussianBlur(radius)).save(a.out/second)
  rows.append(dict(example=i,original_file=first,blurred_file=second,source_image=path.name,radius_pixels=radius,sigma_fraction=.01,cohort='20-image pilot; not the 500-question validation cohorts'))
 if not rows:raise SystemExit('No images found')
 with (a.out/'image_index.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 print('Prepared',len(rows),'pairs')
if __name__=='__main__':main()
