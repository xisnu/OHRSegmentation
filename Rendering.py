from __future__ import print_function
import numpy as np
from PIL import Image,ImageDraw
import h5py

def render_strokes_as_image(allstrokes,filename,box,point_size,corpusline=None):
    maxx = int(box[0])
    minx = int(box[1])
    maxy = int(box[2])
    miny = int(box[3])
    width = maxx - minx + 15
    height = maxy - miny + 15
    img=Image.new("RGB",[width,height],color='white')
    draw=ImageDraw.Draw(img)
    double_size=point_size*2
    for s in allstrokes:
        for p in range(len(s)):
            x=s[p][0]-minx
            y=s[p][1]-miny
            seg_mark=s[p][2]
            draw.ellipse(((x-point_size,y-point_size),(x+point_size,y+point_size)),fill='black')
            if (seg_mark == 2):
                draw.rectangle(((x - double_size, y - double_size), (x + double_size, y + double_size)), fill='green')
            if(corpusline is not None):
                draw.ellipse(((x - point_size, corpusline - point_size), (x + point_size, corpusline + point_size)), fill='blue')
        draw.ellipse(((x - point_size, y - point_size), (x + point_size, y + point_size)), fill='red')
    img.save(filename,"png")

def render_stroke_for_sample(hdffile,sampleid):
    f=h5py.File(hdffile)
    sample=f.get(sampleid)
    nbstrokes = int(sample.attrs["Nb_Strokes"])
    box = sample.attrs["Box"]
    #corpuslineY=float(sample.attrs["CorpuslineY"])
    allstrokes = []
    for strk in range(nbstrokes):
        stroke = np.asarray(sample.get("S" + str(strk)))
        allstrokes.append(stroke)
    render_strokes_as_image(allstrokes, "Samples/temp.png", box, 4)#, corpuslineY)

def convert_online_to_offline(h5file,outputdir):
    f=h5py.File(h5file,"r")
    keys=list(f.keys())
    totaldirs=len(keys)
    for d in range(totaldirs):
        corpuslineY=None
        onesample=f.get(keys[d])
        samplename = onesample.attrs["SampleID"].split(".")[0]
        print("Reading sample ",samplename)
        filename=outputdir+"/"+samplename+".png"
        nbstrokes=int(onesample.attrs["Nb_Strokes"])
        box=onesample.attrs["Box"]
        segmentation=np.asarray(onesample.get('Segmentation'))
        try:
            corpuslineY = float(onesample.attrs["CorpuslineY"])
        except:
            print("No Corpusline Data")
        allstrokes=[]
        for strk in range(nbstrokes):
            stroke=np.asarray(onesample.get("S"+str(strk)))
            allstrokes.append(stroke)
        #print(allstrokes)
        render_strokes_as_image(allstrokes,filename,box,4,corpuslineY)
        #break
    f.close()

sampleid="Aditya11_yAn.txt"
hdffile="/media/parthosarothi/OHWR/Dataset/ICBOHR-W2/Unsegmented/Train_resamp_us.h5"
#render_stroke_for_sample(hdffile,sampleid)
#convert_online_to_offline(hdffile,"Samples/Test")