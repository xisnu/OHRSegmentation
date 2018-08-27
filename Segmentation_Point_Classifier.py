from __future__ import print_function
from PSMUtil import *
import sys
from Segmentation import *
from Rendering import render_strokes_as_image
from PSMFeatures import *
#feat_start=10
def load_data(files):
    green=files[0]
    black=files[1]
    features=[]
    f=h5py.File(green)
    keys=f.keys()
    total=len(keys)
    for k in range(total):
        sample=f.get(keys[k])
        feat=np.asarray(sample.get("Features"))
        # selected=[]
        # for fi in feat_indices:
        #     selected.append(feat[fi])
        #feat=feat[feat_start:]
        features.append([feat,1])
        complete=(k/float(total))*100
        sys.stdout.write("\rPositive sample gathering completed %.2f"%complete)
        sys.stdout.flush()
    f.close()
    print("\nNow reading negative samples")
    f=h5py.File(black)
    keys=f.keys()
    nbsamples=len(keys)
    index=np.random.randint(0,nbsamples,[total])
    for i in index:
        k=keys[i]
        sample=f.get(k)
        feat=np.asarray(sample.get("Features"))
        # selected = []
        # for fi in feat_indices:
        #     selected.append(feat[fi])  #
        #feat=feat[feat_start:]
        features.append([feat,0])
    np.random.shuffle(features)
    x=[]
    y=[]
    print("\nMaking for network")
    for i in range(len(features)):
        inp=features[i][0]
        x.append(inp)
        target=int(features[i][1])
        one_hot=[0,0]
        one_hot[target]=1
        y.append(one_hot)
    return x,y

class Segmentation_Classifier:

    def __init__(self,nb_features,savepath):
        self.nbclasses=2
        self.nbfeatures=nb_features
        self.model=tf.Graph()
        self.savepath=savepath

    def create_network(self,nodes):
        with self.model.as_default():
            self.model_x=tf.placeholder(tf.float32,shape=[None,self.nbfeatures],name="Model_input")
            self.model_y=tf.placeholder(tf.float32,shape=[None,self.nbclasses],name="Model_target")

            layer=self.model_x
            for n in range(len(nodes)):
                layer_name="Dense_"+str(n)
                layer,_=FullyConnected(layer,nodes[n],layer_name)
                layer=tf.tanh(layer)
                shape=get_layer_shape(layer)
                print("Layer ",layer_name," shape ",shape)

            self.logits,self.y_preds=FullyConnected(layer,self.nbclasses,"Output")
            shape=get_layer_shape(self.y_preds)
            print("Output layer shape ",shape)

            #self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.model_y,logits=self.logits))
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars  if 'bias' not in v.name]) * 0.001

            self.loss=tf.reduce_mean(tf.squared_difference(self.model_y,self.y_preds))+lossL2
            self.preds=tf.argmax(self.y_preds,axis=1)
            true=tf.argmax(self.model_y,axis=1)
            corrects=tf.cast(tf.equal(self.preds,true),tf.float32)
            self.accuracy=tf.reduce_mean(corrects)

            self.optimizer=tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)
            #self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)
            self.weight_manager=tf.train.Saver()
            print("Network Ready")

    def train_network(self,traindata,testdata,epochs,batchsize,mode):
        with tf.Session(graph=self.model) as sess:
            if(mode=="Resume"):
                self.weight_manager.restore(sess,self.savepath+"_best")
                print('Previous weights loaded')
            else:
                sess.run(tf.global_variables_initializer())
                print("New weights initialized")
            train_x=traindata[0]
            train_y=traindata[1]
            nbtrains = len(train_x)
            nbbatches = int(np.ceil(nbtrains / float(batchsize)))
            print("Train on %d data"%nbtrains)
            if(testdata is not None):
                test_x=testdata[0]
                test_y=testdata[1]
                nbtests = len(test_x)
                nbtest_batches=int(np.ceil(nbtests/float(batchsize)))
                print("Test on %d data" % nbtests)
            for e in range(epochs):
                start=0
                loss=0
                acc=0
                best_acc=0
                for b in range(nbbatches):
                    end=min(nbtrains-1,start+batchsize)
                    batch_x=train_x[start:end]
                    batch_y=train_y[start:end]
                    feed={self.model_x:batch_x,self.model_y:batch_y}
                    l,a,_=sess.run([self.loss,self.accuracy,self.optimizer],feed_dict=feed)
                    sys.stdout.write("\rReading batch from %d to %d , Loss %f, Accuracy %f"%(start,end,l,a))
                    sys.stdout.flush()
                    loss+=l
                    acc+=a
                    start=end
                loss/=nbbatches
                acc/=nbbatches
                self.weight_manager.save(sess,self.savepath+"_last")
                print("\nEpoch %d Training loss %f Accuracy %f" % (e,loss, acc))
                if(testdata is not None):
                    t_start = 0
                    t_loss = 0
                    t_acc = 0
                    for b in range(nbtest_batches):
                        t_end = min(nbtests - 1, t_start + batchsize)
                        batch_x = test_x[t_start:t_end]
                        batch_y = test_y[t_start:t_end]
                        feed = {self.model_x: batch_x, self.model_y: batch_y}
                        l, a= sess.run([self.loss, self.accuracy], feed_dict=feed)
                        #sys.stdout.write("\rReading test batch from %d to %d , Loss %f, Accuracy %f" % (t_start, t_end, l, a))
                        t_loss += l
                        t_acc += a
                        t_start=t_end
                    t_loss /= nbtest_batches
                    t_acc /= nbtest_batches
                    if(best_acc<t_acc):
                        best_acc=t_acc
                        self.weight_manager.save(sess,self.savepath+"_best")
                    print("\tTesting loss %f Accuracy %f Best Accuracy=%f" % (t_loss, t_acc,best_acc))


    def predict(self,features):
        inp=[features]
        with tf.Session(graph=self.model) as sess:
            self.weight_manager.restore(sess, self.savepath)
            output=sess.run(self.preds,feed_dict={self.model_x:inp})
        #print(output)
        return output

    def perform_random_tests(self,hdffile, nbtests,vicinity_half):
        vicinity_length=2*vicinity_half+1
        f = h5py.File(hdffile)
        keys = f.keys()
        total = len(keys)
        with tf.Session(graph=self.model) as sess:
            self.weight_manager.restore(sess, self.savepath+"_best")
            print("Weights are loaded")
            for nt in range(nbtests):
                ind = np.random.randint(0, total)
                sample = f.get(keys[ind])
                sampleid = sample.attrs['SampleID']
                nbstrokes = int(sample.attrs['Nb_Strokes'])
                corpuslineY = float(sample.attrs['CorpuslineY'])
                box = sample.attrs["Box"]
                print("processing sample %s with %d strokes"%(sampleid,nbstrokes))
                all_strokes = []
                min_d_from_cl = 1000
                for n in range(nbstrokes):
                    a_stroke = np.asarray(sample.get("S" + str(n)))
                    stroke_length = total_stroke_length(a_stroke)
                    nb_points = len(a_stroke)
                    i = vicinity_half
                    start=0
                    while (i < nb_points - (vicinity_half+1)):
                        region = a_stroke[i - vicinity_half:i + vicinity_half+1]
                        central_point = region[vicinity_half]
                        features = get_green_point_feature(region, corpuslineY, a_stroke, stroke_length)
                        out = sess.run(self.preds, feed_dict={self.model_x: [features]})
                        if (out[0] == 1):
                            #dist_from_cl = corpuslineY - central_point[1]
                            # if (dist_from_cl < min_d_from_cl):
                            #     min_d_from_cl = dist_from_cl
                            cut_len=i-start
                            if(cut_len>vicinity_length):
                                a_stroke[i][2] = 2
                                i += 10
                                start=i
                        i += 7
                        all_strokes.append(a_stroke)
                    render_strokes_as_image(all_strokes, "Samples/Predicted/" + sampleid + "_pr.png", box, 4,corpusline=corpuslineY)
                print("Image Ready %d/%d"%(nt,nbtests))

    def separate_strokes_for_cluster(self,hdfin,hdfout):
        f = h5py.File(hdfin)
        f2=h5py.File(hdfout,"w")
        keys = f.keys()
        total = len(keys)
        with tf.Session(graph=self.model) as sess:
            self.weight_manager.restore(sess, self.savepath + "_best")
            print("Weights are loaded")
            for ind in range(total):
                sample = f.get(keys[ind])
                # valid=sample.attrs['Valid']
                # if(valid=='No'):
                #     continue
                sampleid = sample.name
                nbstrokes = int(sample.attrs['Nb_Strokes'])
                #box = sample.attrs["Global_Box"]
                corpuslineY = float(sample.attrs['CorpuslineY'])
                #corpuslineY=float(box[-1])
                print("processing sample %s with %d strokes" % (sampleid, nbstrokes))
                min_d_from_cl = 1000
                for n in range(nbstrokes):
                    strokename="S" + str(n)
                    a_stroke = np.asarray(sample.get(strokename))
                    try:
                        stroke_length = total_stroke_length(a_stroke)
                        nb_points = len(a_stroke)
                        i = 3
                        start=0
                        while (i < nb_points - 4):
                            region = a_stroke[i - 3:i + 4]
                            central_point = region[3]
                            features = get_green_point_feature(region, corpuslineY, a_stroke, stroke_length)
                            out = sess.run(self.preds, feed_dict={self.model_x: [features]})
                            if (out[0] == 1): #Found one green point
                                # dist_from_cl = corpuslineY - central_point[1]
                                # if (dist_from_cl < min_d_from_cl):
                                #     min_d_from_cl = dist_from_cl
                                end=i
                                cut_stroke=a_stroke[start:end+1]
                                if(len(cut_stroke)>7):
                                    gname=sampleid+"_"+strokename+"_"+str(i)
                                    g=f2.create_group(gname)
                                    g.attrs['SampleID']=sampleid
                                    #g.attrs['Part_of']=sample.attrs['SS_Label']
                                    g.attrs['Stroke_Length']=len(cut_stroke)
                                    #g.attrs['Annotation']=sample.attrs['Annotation']
                                    g.create_dataset("Stroke",data=cut_stroke)
                                    print("\tCreating group %s stroke length %d"%(gname,len(cut_stroke)))
                                    start=end+1
                                    i += 7
                            i += 10
                    except:
                        print("Exception for ",sampleid)
                        pass
            f.close()
            f2.close()


nodes=[32,16,8]
def fit_model(mode):#Do not use
    path="/media/parthosarothi/OHWR/Dataset/ICBOHR-W2/Unsegmented"
    trainfiles=[path+"/Green_Features_Train.h5",path+"/Black_Features_Train.h5"]
    testfiles=[path+"/Green_Features_Test.h5",path+"/Black_Features_Test.h5"]
    train_x,train_y=load_data(trainfiles)
    test_x,test_y=load_data(testfiles)
    nb_features=len(train_x[0])
    print("Features=%d"%nb_features)
    net=Segmentation_Classifier(nb_features,"Weights/Segmentation_Classifier")
    net.create_network(nodes)
    net.train_network([train_x,train_y],[test_x,test_y],500,256,mode)


def predict_points_of_sample(nb_features,sampleid,hdffile):
    net = Segmentation_Classifier(nb_features, "Weights/Segmentation_Classifier_best")
    net.create_network(nodes)
    f=h5py.File(hdffile)
    sample=f.get(sampleid)
    nbstrokes=int(sample.attrs['Nb_Strokes'])
    corpuslineY=float(sample.attrs['CorpuslineY'])
    box=sample.attrs["Box"]
    all_strokes=[]
    min_d_from_cl=1000
    for n in range(nbstrokes):
        a_stroke=np.asarray(sample.get("S"+str(n)))
        stroke_length=total_stroke_length(a_stroke)
        nb_points=len(a_stroke)
        i=3
        while(i<nb_points-4):
            region=a_stroke[i-3:i+4]
            central_point=region[3]
            features=get_green_point_feature(region,corpuslineY,a_stroke,stroke_length)
            #print(features)
            out=net.predict(features)
            if(out[0]==1):
                dist_from_cl=corpuslineY-central_point[1]
                if(dist_from_cl<min_d_from_cl):
                    min_d_from_cl=dist_from_cl
                a_stroke[i][2]=2
                i+=10
            i+=1
        all_strokes.append(a_stroke)
    render_strokes_as_image(all_strokes,"Samples/"+sampleid+"_predicted.png",box,2,corpusline=corpuslineY)
    print("Image Ready")

def test_some_samples(hdffile,nbtests,nb_features):
    net = Segmentation_Classifier(nb_features, "Weights/Segmentation_Classifier")
    net.create_network(nodes)
    net.perform_random_tests(hdffile,nbtests,3)

def generate_substroke_for_cluster(nbfeat,hdffile,hdfout):
    net = Segmentation_Classifier(nbfeat, "Weights/Segmentation_Classifier")
    net.create_network(nodes)
    net.separate_strokes_for_cluster(hdffile,hdfout)


nbfeat=12#Do not change
hdfin="Data/Train_resamp_us.h5"
hdfout="Data/Unlabelled_Segmented_Train.h5"

#fit_model("Resume")

#predict_points_of_sample(nbfeat,"Akinchan34_AmdAnI.txt",hdfin)

#test_some_samples(hdffile,100,nbfeat)

generate_substroke_for_cluster(nbfeat,hdfin,hdfout)


