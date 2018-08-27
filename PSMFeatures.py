from __future__ import print_function
import numpy as np
#from Offline_Processing import *
'''
Online Context [For every point P(x,y)]
1. Pen tip touching board or not (0)
2. Hat feature for delayed stroke (0)
3. velocity before resampling (0)
4. x corordinate after high-pass filtering using moving average (1)
5. y coordinate after normalization (0)
6. cosine and sine of angle between line segment starting at P and X Axis (2)
7. cosine and sine of angle between lines to the previous and next point (2)
8. vicinity aspect (1)
9. cosine and sine of angle between the straight line connecting first and last point of vicinity and X asis (2)
10 length of trajectory divided by Max(delx,dely) (1)
11. average square distance of each point to the straight line (1)
12. theta1, theta2 are slope of lines from Pt-1 to Pt and Pt to Pt+1, returns cos and sin of (theta2-theta1)
13. theta1, theta2 are slope of lines from P0 to Pt and Pt to Pn-1, returns cos and sin of (theta2-theta1) for a vivinity of length n (2)
14. give direction codes [NE,NW,SE,SW] from start to center and from center to end of a vicinity (2)
15. normalized distance of start,center,end point to corpusline (3)
16. total stroke length
17. normalized distance of central point from the beginning and end of stroke (2)
'''
def finddistance(fp,sp):
    x1=fp[0]
    y1=fp[1]
    x2=sp[0]
    y2=sp[1]
    dx=(x2-x1)**2
    dy=(y2-y1)**2
    dist=np.sqrt(dx+dy)
    return dist

def slope_of_line(fp,sp):
    del_x=sp[0]-fp[0]
    del_y=sp[1]-fp[1]
    hyp=np.sqrt((del_x**2)+(del_y**2))
    if(del_x==0):
        cos_theta=0
        sin_theta=1.57
    else:
        cos_theta=del_x/hyp
        sin_theta=del_y/hyp
    return cos_theta,sin_theta

def distance_from_line(line,point):
    a=line[0]
    b=line[1]
    c=line[2]
    x=point[0]
    y=point[1]
    num=abs(a*x+b*y+c)
    den=np.sqrt(a**2+b**2)
    dist=num/den
    return dist

#OK
def direction_from_point(from_point,to_point):
    #Finds the position of to_point from from_point (NE,NW,SE,SW)
    fx=from_point[0]
    fy=from_point[1]
    tx=to_point[0]
    ty=to_point[1]
    if(fx<=tx) and (fy>ty):#NE
        pos=0.25
    elif(fx>tx) and (fy>=ty):#NW
        pos=0.5
    elif(fx>=tx) and (fy<ty):#SW
        pos=0.75
    elif(fx<tx) and (fy<=ty):#SE
        pos=1.0
    return pos

#OK
def angle_between_points(fp,sp):
    base=abs(sp[0]-fp[0])
    height=sp[1]-fp[1]
    hyp=np.sqrt(base**2+height**2)+0.00001
    cos_theta=base/hyp
    sin_theta=np.sqrt(1-cos_theta**2)
    #print(base,height)
    theta=[cos_theta,sin_theta]
    return theta

#0
def feature_4(vicinity): # 0
    totalpoints=len(vicinity)#Must be ODD
    xs=[vicinity[t][0] for t in range(totalpoints)]
    moving_average=np.mean(xs)
    mid_index=int(totalpoints/2.0)
    feat_4=xs[mid_index]-moving_average
    #print("X values ", xs," index ",mid_index," Feature value=",feat_4)
    return [feat_4]

#Ok
def feature_5(vicinity,box):
    #y coordinate after normalization (1)
    maxy=box[2]
    miny=box[3]
    height=maxy-miny
    totalpoints = len(vicinity)  # Must be ODD
    mid_index = int(totalpoints / 2.0)
    y=vicinity[mid_index][1]
    feat_5=y/float(height)
    #print("Height=",height," Index",mid_index," Feat=",feat_5)
    return [feat_5]

#Ok 1 2
def feature_6(vicinity):
    #cosine and sine of angle between line segment starting at P and X Axis (2)
    mid_index = int(len(vicinity) / 2.0)
    x=vicinity[mid_index][0]
    y=vicinity[mid_index][1]
    x_next=vicinity[mid_index+1][0]
    y_next=vicinity[mid_index+1][1]
    #feat_6=angle_between_points([x,y],[x_next,y_next])
    hyp=np.sqrt((x_next-x)**2+(y_next-y)**2)
    costheta=(x_next-x)/hyp
    sintheta=np.sqrt(1-costheta**2)
    #print("Feat-6=",feat_6," Index=",mid_index)
    return [costheta,sintheta]

#OK 3 4
def feature_7(vicinity):
    #cosine and sine of angle between lines to the previous and next point (2)
    mid_index = int(len(vicinity) / 2.0)
    x = vicinity[mid_index][0]
    y = vicinity[mid_index][1]
    x_next = vicinity[mid_index + 1][0]
    y_next = vicinity[mid_index + 1][1]
    x_prev=vicinity[mid_index - 1][0]
    y_prev=vicinity[mid_index - 1][1]
    a_dot_b=((x_prev-x)*(x_next-x))+((y_prev-y)*(y_next-y))
    magnitude=np.sqrt((x_prev-x)**2+(y_prev-y)**2)*np.sqrt((x_next-x)**2+(y_next-y)**2)
    cos_theta=a_dot_b/magnitude
    cos_theta2=cos_theta**2
    sin_theta2=1.0-(cos_theta2)
    sin_theta=np.sqrt(abs(sin_theta2))
    #print(cos_theta,sin_theta)
    return [cos_theta,sin_theta]

#OK 5
def feature_8(vicinity):
    # vicinity aspect
    del_y=abs(vicinity[-1][1]-vicinity[0][1])
    del_x = abs(vicinity[-1][0] - vicinity[0][0])
    try:
        feat_8=(del_y-del_x)/float(del_y+del_x)
    except:
        print("Circular curve warning")
        feat_8=0
    #print(feat_8)
    return [feat_8]

#OK 6 7
def feature_9(vicinity):
    #cosine and sine of angle between line connecting first point and last point and X axis, returns 2 values
    fp=vicinity[0]
    sp=vicinity[-1]
    x1=fp[0]
    y1=fp[1]
    x2=sp[0]
    y2=sp[1]
    try:
        hyp=np.sqrt((x2-x1)**2+(y2-y1)**2)
        costheta=(x2-x1)/hyp
        sintheta=(y2-y1)/hyp
        feat_9=[costheta,sintheta]
    except:
        print(fp,sp,hyp,vicinity)
    #print(feat_9)
    return feat_9

# 8
def feature_10(vicinity):
    #Find length of trajectory divided by Max(delx,dely) returns 1 value
    totalpoints=len(vicinity)
    trajectory_length=0
    for t in range(1,totalpoints):
        fp=vicinity[t-1]
        sp=vicinity[t]
        dist=(sp[0]-fp[0])**2+(sp[1]-fp[1])**2
        sq_dist=np.sqrt(dist)
        trajectory_length+=sq_dist
    delx = abs(vicinity[-1][0] - vicinity[0][0])
    dely = abs(vicinity[-1][1] - vicinity[0][1])
    divide_by=max([delx,dely])
    try:
        feat_10=trajectory_length/float(divide_by)
    except:
        print("Circular Curve Warning")
        feat_10 = trajectory_length / 0.001
    return [feat_10]

# 9
def feature_11(vicinity):
    #average square distance of each point from straight line connecting first and last point, retuns 1 value
    totalpoints=len(vicinity)
    #print("\n", vicinity[0], vicinity[-1], "\n")
    del_x=float(vicinity[-1][0]-vicinity[0][0])
    if(del_x==0):
        m=57.8
    else:
        m=(vicinity[-1][1]-vicinity[0][1])/del_x
    c=vicinity[0][1]-m*vicinity[0][0]
    line=[m,-1,c]
    total_dist=0
    for t in range(totalpoints):
        point=vicinity[t]
        dist=distance_from_line(line,point)
        total_dist+=dist
    feat_11=total_dist/totalpoints
    return [feat_11]
# 10
def feature_12(vicinity):
    total = len(vicinity)
    center_index = total / 2
    central_point = vicinity[center_index]
    gp_m1 = vicinity[center_index - 1]  # a point before green point
    gp_p1 = vicinity[center_index + 1]  # a point after green point
    cos1, sin1 = slope_of_line(gp_m1, central_point)
    cos2, sin2 = slope_of_line(central_point, gp_p1)
    diff_cos = cos2 - cos1
    diff_sin = sin2 - sin1
    return [diff_cos,diff_sin]

# 11 12
def feature_13(vicinity):
    total = len(vicinity)
    center_index = total / 2
    central_point = vicinity[center_index]
    gp_m1 = vicinity[0]  # starting point
    gp_p1 = vicinity[-1]  # terminal point
    cos1, sin1 = slope_of_line(gp_m1, central_point)
    cos2, sin2 = slope_of_line(central_point, gp_p1)
    diff_cos = cos2 - cos1
    diff_sin = sin2 - sin1
    return [diff_cos,diff_sin]

# 13 14
def feature_14(vicinity):
    total = len(vicinity)
    center_index = total / 2
    central_point = vicinity[center_index]
    start=vicinity[0]
    dir_start_to_center=direction_from_point(start,central_point)
    end=vicinity[-1]
    dir_center_to_end=direction_from_point(central_point,end)
    return [dir_start_to_center,dir_center_to_end]

# 15 16 17
def feature_15(vicinity,corpuslineY):
    total = len(vicinity)
    center_index = total / 2
    central_point = vicinity[center_index]
    start = vicinity[0]
    end = vicinity[-1]
    dist1=finddistance(start,[start[0],corpuslineY])
    dist2=finddistance(central_point,[central_point[0],corpuslineY])
    dist3=finddistance(end,[end[0],corpuslineY])
    max_dist=float(max([dist1,dist2,dist3]))
    feat=[dist1/max_dist,dist2/max_dist,dist3/max_dist]
    return feat

def total_stroke_length(stroke):# Find total length of stroke
    total=len(stroke)-1
    total_len=0
    for t in range(total):
        fp=stroke[t]
        sp=stroke[t+1]
        d=finddistance(fp,sp)
        total_len+=d
    return total_len

# 18 19
def feature_17(vicinity,stroke,total_stroke_len):
    total = len(vicinity)
    center_index = total / 2
    central_point = vicinity[center_index]
    fp_stroke=stroke[0]
    lp_stroke=stroke[-1]
    d1=finddistance(fp_stroke,central_point)/float(total_stroke_len)
    d2=finddistance(central_point,lp_stroke)/float(total_stroke_len)
    return [d1,d2]
#length of vicinity -1
def fft_of_vicinity(vicinity):
    vicinity=np.asarray(vicinity)
    xs=vicinity[:,0]
    ys=vicinity[:,1]
    x_diff=[xs[t-1]-xs[t] for t in range(1,len(xs))]
    y_diff = [ys[t - 1] - ys[t] for t in range(1, len(ys))]
    x_fft=abs(np.fft.fft(x_diff))
    y_fft = abs(np.fft.fft(y_diff))
    max_x_val=max(x_fft)+1
    max_y_val = max(y_fft) + 1
    x_norm_fft=np.divide(x_fft,max_x_val)
    y_norm_fft = np.divide(y_fft, max_y_val)
    norm_fft=[]
    norm_fft.extend(x_norm_fft)
    norm_fft.extend(y_norm_fft)
    return norm_fft

def gather_all_features_from_vicinity(vicinity,box,corpuslineY,stroke,total_stroke_len):
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
    feat = feature_15(vicinity, corpuslineY)
    features.extend(feat)
    feat = feature_17(vicinity,stroke,total_stroke_len)
    features.extend(feat)
    feat=fft_of_vicinity(vicinity)
    features.extend(feat)
    return features

def compute_feature_for_sample(allstrokes,box,vicinity_length,shift):
    total=len(allstrokes)
    vicinity_half=vicinity_length/2
    stroke_features=[]
    nb_feat=34
    for t in range(total):
        a_stroke=allstrokes[t]
        tsl = total_stroke_length(a_stroke)
        corpusliney = find_corpusline_from_strokes(allstrokes, box, "corpusline.png")
        total_points=len(a_stroke)
        if(total_points>=vicinity_length):
            i=vicinity_half
            while(i<total_points-vicinity_half-2):
                vicinity=a_stroke[i-vicinity_half:i+vicinity_half+1]
                features=gather_all_features_from_vicinity(vicinity,box,corpusliney,a_stroke,tsl)
                stroke_features.append(features)
                #print("\t\tFeatures found for %d  = %d"%(i,len(features)))
                i=i+shift
            #nb_feat=len(features)
            eos_features=np.zeros(nb_feat)
            eos_features.fill(-2)
            #print("\t\tFeatures found for EOS  = %d" % (len(features)))
            stroke_features.append(eos_features)

    return np.asarray(stroke_features)

feat_indices=[0,1,2,3,4,6,7,10,11,12,13,14,15,16,17,18,19]

# fp=[50,100]
# tp=[55,100]
# pos=direction_from_point(fp,tp)
# print(pos)