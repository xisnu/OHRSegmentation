from __future__ import print_function
import numpy as np
import h5py,traceback
from PSMFeatures import *
#red rectangles represent natural end of stroke
#green rectangles represents manual annotation

def check_how_many_green_samples(hdffile):
    f=h5py.File(hdffile)
    keys=f.keys()
    total=len(keys)
    green_count=0
    sample_count=0
    f2=open("Mannually_Annotated.txt","w")
    for t in range(total):
        sample=f.get(keys[t])
        sampleid=sample.attrs['SampleID']
        nbstrokes=int(sample.attrs['Nb_Strokes'])
        flag=False
        this_gcount=0
        for s in range(nbstrokes):
            strokename='S'+str(s)
            astroke=np.asarray(sample.get(strokename))
            for p in range(len(astroke)):
                seg_mark=astroke[p][-1]
                if(seg_mark==2):
                    flag=True
                    this_gcount+=1
        if(flag):
            f2.write(sampleid+","+str(this_gcount)+"\n")
            sample_count+=1
            green_count+=this_gcount
        print('Reading sample %s , strokes %d, demarks %d'%(sampleid,nbstrokes,this_gcount))
    print("Mannual Annotation Available %d , Samples %d"%(green_count,sample_count))
    f.close()
    f2.close()

def find_green_region_from_sample(hdffile,sampleid):
    f=h5py.File(hdffile)
    sample=f.get(sampleid)
    segmarks=np.asarray(sample.get("Segmentation"))
    nbstrokes=sample.attrs['Nb_Strokes']
    stroke_lengths=[]
    all_strokes=[]
    for n in range(nbstrokes):
        strokename="S"+str(n)
        a_stroke=sample.get(strokename)
        nbpoints=len(a_stroke)
        stroke_lengths.append(nbpoints)
        all_strokes.append(a_stroke)
    region=[]
    for l in range(len(segmarks)):
        if(segmarks[l]==2):
            acc=0
            s=0
            for s in range(nbstrokes):
                acc+=stroke_lengths[s]
                if(acc>l):
                    break
            prev_stroke=s-1
            print("Green at ", l," between stroke %d and %d"%(prev_stroke,s))
            region.append(all_strokes[prev_stroke])
            region.append(all_strokes[s])
    f.close()
    return region

def get_green_point_feature(region,corpuslineY,stroke,stroke_len):
    #print("Region length=",len(region))
    features=[]
    feat_11=feature_11(region)
    features.extend(feat_11)
    #print("11 length is ", len(features))
    feat_12 = feature_12(region)
    features.extend(feat_12)
    #print("12 length is ", len(features))
    feat_13 = feature_13(region)
    features.extend(feat_13)
    #print("13 length is ", len(features))
    feat_14 = feature_14(region)
    features.extend(feat_14)
    #print("14 length is ", len(features))
    feat_15 = feature_15(region,corpuslineY)
    features.extend(feat_15)
    #print("15 length is ", len(features))
    feat_17 = feature_17(region,stroke,stroke_len)
    features.extend(feat_17)
    #print("17 length is ", len(features))
    return features

def get_green_features_from_stroke(stroke,gp_index,half_window,corpuslineY):
    stroke_length=total_stroke_length(stroke)
    region=stroke[gp_index-half_window:gp_index+half_window+1]
    features=get_green_point_feature(region,corpuslineY,stroke,stroke_length)
    return features

def collect_all_green_points(hdfin,hdfout):
    f=h5py.File(hdfin)
    f2=h5py.File(hdfout,"w")
    f_ex=open("Exception.txt","w")
    keys=f.keys()
    for k in keys:
        sample=f.get(k)
        sampleid=sample.attrs['SampleID']
        corpuslineY=float(sample.attrs['CorpuslineY'])
        nbstrokes=int(sample.attrs['Nb_Strokes'])
        flag="Not Found"
        for s in range(nbstrokes):
            strokename="S"+str(s)
            a_stroke=np.asarray(sample.get(strokename))
            for i in range(len(a_stroke)):
                segmark=a_stroke[i][2]
                if(segmark==2):
                    try:
                        feat=get_green_features_from_stroke(a_stroke,i,3,corpuslineY)
                        #print(feat)
                        group_name=sampleid+"_"+str(s)+"_"+str(i)
                        print("\tCreating group ",group_name)
                        g=f2.create_group(group_name)
                        g.create_dataset("Features",data=feat)
                        flag="Found"
                    except:
                        f_ex.write(sampleid+","+strokename+"\n")
                        pass
                        #break

        print("Processing %s for positive samples , %s"%(sampleid,flag))
    f.close()
    f2.close()
    f_ex.close()

def collect_some_black_points(hdfin,hdfout):
    f=h5py.File(hdfin)
    f2=h5py.File(hdfout,"w")
    keys=f.keys()
    for k in keys:
        sample=f.get(k)
        sampleid=sample.attrs['SampleID']
        corpuslineY=float(sample.attrs['CorpuslineY'])
        nbstrokes=int(sample.attrs['Nb_Strokes'])
        flag=False
        for s in range(nbstrokes):
            strokename="S"+str(s)
            a_stroke=np.asarray(sample.get(strokename))
            nbpoints=len(a_stroke)
            for i in range(nbpoints):
                segmark=a_stroke[i][2]
                if(segmark==2):
                    flag=True
                    break
            if(not flag)and(nbpoints>7):
                try:
                    pt_index=np.random.randint(0,nbpoints-7)
                    feat=get_green_features_from_stroke(a_stroke,pt_index,3,corpuslineY)
                    group_name=sampleid+"_"+str(s)+"_"+str(pt_index)
                    #print("\tCreating group ",group_name)
                    g=f2.create_group(group_name)
                    g.create_dataset("Features",data=feat)
                    flag=False
                except:
                    pass
        print("Processing %s for negative samples , %s"%(sampleid,flag))
    f.close()
    f2.close()

def gather_online_features_from_vicinity(vicinity,box,stroke,total_stroke_len):
    features=[]
    feat=feature_4(vicinity)
    features.extend(feat)
    feat=feature_5(vicinity,box)
    features.extend(feat)
    feat = feature_6(vicinity)
    features.extend(feat)
    feat = feature_7(vicinity)
    features.extend(feat)
    feat = feature_8(vicinity)
    features.extend(feat)
    feat = feature_9(vicinity)
    features.extend(feat)
    feat = feature_10(vicinity)
    features.extend(feat)
    feat = feature_11(vicinity)
    features.extend(feat)
    feat = feature_12(vicinity)
    features.extend(feat)
    feat = feature_13(vicinity)
    features.extend(feat)
    feat = feature_14(vicinity)
    features.extend(feat)
    feat = feature_17(vicinity,stroke,total_stroke_len)
    features.extend(feat)
    feat=fft_of_vicinity(vicinity)
    features.extend(feat)
    return features

def compute_online_feature_for_sample(allstrokes,box,vicinity_length,shift):
    total=len(allstrokes)
    vicinity_half=vicinity_length/2
    stroke_features=[]
    #nb_feat=34
    for t in range(total):
        a_stroke=allstrokes[t]
        tsl = total_stroke_length(a_stroke)
        total_points=len(a_stroke)
        if(total_points>=vicinity_length):
            i=vicinity_half
            while(i<total_points-vicinity_half-2):
                vicinity=a_stroke[i-vicinity_half:i+vicinity_half+1]
                features=gather_online_features_from_vicinity(vicinity,box,a_stroke,tsl)
                stroke_features.append(features)
                #print("\t\tFeatures found for %d  = %d"%(i,len(features)))
                i=i+shift

    return np.asarray(stroke_features)

def get_feature_from_ss_cluster(hdfin,hdfout,vicinity):
    f=h5py.File(hdfin)
    keys=f.keys()
    for k in keys:
        sample=f.get(k)
        stroke=np.asarray(sample.get("Stroke"))
        xs=stroke[:,0]
        ys=stroke[:1]
        maxx=max(xs)
        minx=min(xs)
        maxy=max(ys)
        miny=min(ys)
        box=[maxx,minx,maxy,miny]
        print(box)
        if(len(stroke)>=vicinity):
            feat=compute_online_feature_for_sample([stroke],box,vicinity,3)
            print('Reading stroke ',sample.name," feature ",feat.shape)


path="/media/parthosarothi/OHWR/Dataset/ICBOHR-W2/Unsegmented/"
hdffile="/media/parthosarothi/OHWR/Dataset/ICBOHR-W2/Unsegmented/Test_resamp_us.h5"
hdfout="/media/parthosarothi/OHWR/Dataset/ICBOHR-W2/Unsegmented/Black_Features_Train.h5"

#check_how_many_green_samples(hdffile)
#region=find_green_region_from_sample(hdffile,"Subhrasundargoswami13_paramANu.txt")
#collect_all_green_points(hdffile,hdfout)
#collect_some_black_points(hdffile,hdfout)
#get_feature_from_ss_cluster(path+"/SS_Cluster.h5","",7)


