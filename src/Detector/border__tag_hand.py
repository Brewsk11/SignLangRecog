from PIL import Image
import os
# def get_pixel(image, i, j):
#   # Inside image bounds?
#   width, height = image.size
#   if i > width or j > height:
#     return None

#   # Get Pixel
#   pixel = image.getpixel((i, j))
#   return pixel

def get_hand_border(img):
  width, height = img.size
  minx, miny = width, height
  maxx = maxy = 0
  for i in range(height):
    for j in range(width):
      if img.getpixel((j,i))[0] == 255:  #pixel is white
        if i < miny:
          miny = i
        elif i > maxy:
          maxy = i
        
        if j < minx:
          minx = j
        elif j > maxx:
          maxx = j

  if minx==0: minx+=1
  if miny==0: miny+=1
  if maxx==width: maxx-=1
  if maxy==height: maxy-=1
  return minx, miny, maxx, maxy

def get_xml(folder, filename, path, width, height, xmin, ymin, xmax, ymax):
  string = '''\
<annotation>
	<folder>{folder}</folder>
	<filename>{filename}</filename>
	<path>{path}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>hand</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
</annotation>\
  '''.format(folder=folder, filename=filename, path=path, width=width, height=height, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
  return string

def zoom(img):
  width, height = img.size  
  
  left = 3
  top = 3
  right = width - 3
  bottom = height - 3
    
  im1 = img.crop((left, top, right, bottom)) 
  newsize = (width, height) 
  im1 = im1.resize(newsize) 

  return im1

root_path = "Literki tagged/"
folders = []
for r,d,f in os.walk(root_path):
  for directory in d:
    folders.append(os.path.join(r,directory))

for folder in folders:
  dir_name = os.path.basename(folder).replace("_Tagged","")
  os.mkdir("border_tag/"+dir_name)
  os.mkdir("Transform/"+dir_name)
  for r,d,files in os.walk(folder):
    for file in files:
      # if "128.bmp" in file:
      img = Image.open(os.path.join(os.getcwd(), folder, file))
      img = zoom(img)
      img_name = file
      img.save("Transform\\"+dir_name+"\\"+img_name)
      xmin, ymin, xmax, ymax = get_hand_border(img)
      size = file.split("_")[-1].replace(".bmp","")
      xml = get_xml(os.path.join(os.getcwd(),folder), file.replace("bmp","jpg"), os.path.join(os.getcwd(), folder, file.replace("bmp","jpg")), size, size, xmin, ymin, xmax, ymax)
      fstream = open("border_tag/"+dir_name+"/"+file.replace("bmp","xml"), "w")
      fstream.write(xml)
      fstream.close()
  print(folder)
