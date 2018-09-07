from __future__ import print_function
from PyQt4 import QtGui, QtCore
import sys,h5py
from PyQt4.QtGui import *
import numpy as np

def trace_tree_path(item,path):
    parent=item.parent()
    if(parent is not None):
        #print(path)
        trace_tree_path(parent,path)
        path.append(str(parent.text(0)))
    else:
        return path


def walk_hdf(hdf_element,name,subtree):
    if (type(hdf_element) == h5py.Group):
        subtree=QtGui.QTreeWidgetItem(subtree,[name])
        keys=hdf_element.keys()
        for k in keys:
            s = hdf_element.get(k)
            walk_hdf(s,k,subtree)
    else:
        QtGui.QTreeWidgetItem(subtree,[name])

def explore_hdf_n_populate_tree(filename,tree):
    f = h5py.File(filename)
    keys=f.keys()
    for k in keys:
        s=f.get(k)
        walk_hdf(s,k,tree)


class Window(QtGui.QWidget):
    def __init__(self, box):
        QtGui.QWidget.__init__(self)
        self.setWindowTitle("HDF Viewer")
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        layout_main = QtGui.QVBoxLayout(self)
        layout_main.setGeometry(QtCore.QRect(box[0], box[1], box[2], box[3]))
        self.width=box[2]-box[0]
        self.height=box[3]-box[1]
        self.move(0,0)

        #browsing file area
        layout_browse=QtGui.QHBoxLayout(self)
        btn_browse_hdf=QtGui.QPushButton("Open")
        btn_browse_hdf.clicked.connect(self.browse_hdf)
        layout_browse.addWidget(btn_browse_hdf)
        self.txt_filename=QtGui.QTextEdit(self)
        self.txt_filename.setFixedHeight(30)
        self.txt_filename.setFixedWidth(self.width*0.75)
        layout_browse.addWidget(self.txt_filename)
        layout_main.addLayout(layout_browse)

        #file list area
        layout_fileinfo=QtGui.QHBoxLayout(self)
        layout_filecontent = QtGui.QVBoxLayout(self)
        self.lst_filecontent = QtGui.QTableWidget()
        self.lst_filecontent.setFixedWidth(self.width * 0.20)
        self.lst_filecontent.setFixedHeight(self.height * 0.20)
        layout_filecontent.addWidget(self.lst_filecontent)

        self.tre_filelist = QtGui.QTreeWidget()
        self.tre_filelist.setFixedHeight(self.height * 0.40)
        self.tre_filelist.setFixedWidth(self.width * 0.20)
        self.tre_filelist.clicked.connect(self.tre_filelist_onclick)
        layout_filecontent.addWidget(self.tre_filelist, QtCore.Qt.AlignLeft)
        layout_fileinfo.addLayout(layout_filecontent)
        self.display_canvas=QtGui.QGraphicsScene(self)
        self.display_area=QtGui.QGraphicsView(self)
        self.display_area.setScene(self.display_canvas)
        self.display_area.setFixedWidth(self.width*0.40)
        self.display_area.setFixedHeight(self.height*0.60)
        layout_fileinfo.addWidget(self.display_area,QtCore.Qt.AlignTop)
        self.tab_dataset_values=QtGui.QTableWidget(self)
        self.tab_dataset_values.setFixedWidth(self.width*0.30)
        self.tab_dataset_values.setFixedHeight(self.height*0.60)
        layout_fileinfo.addWidget(self.tab_dataset_values)
        layout_main.addLayout(layout_fileinfo)

        layout_status=QtGui.QHBoxLayout(self)
        self.lbl_current_sample=QtGui.QLabel(self)
        layout_status.addWidget(self.lbl_current_sample)
        self.lbl_current_key=QtGui.QLabel(self)
        layout_status.addWidget(self.lbl_current_key)
        btn_display_stroke=QtGui.QPushButton("Display As Stroke")
        btn_display_stroke.clicked.connect(self.display_multiple_stroke)
        layout_status.addWidget(btn_display_stroke)
        btn_append_stroke=QtGui.QPushButton("Add to Display")
        btn_append_stroke.clicked.connect(self.append_stroke)
        layout_status.addWidget(btn_append_stroke)
        btn_refresh_canvas=QtGui.QPushButton("Refresh Canvas")
        btn_refresh_canvas.clicked.connect(self.refresh_canvas)
        layout_status.addWidget(btn_refresh_canvas)
        layout_main.addLayout(layout_status)

        self.current_path=[]
        self.current_data=[]
        self.colorpallete=[QtCore.Qt.red,QtCore.Qt.green,QtCore.Qt.blue,QtCore.Qt.magenta,QtCore.Qt.yellow,QtCore.Qt.blue]
        self.current_color_index=0
        self.current_stroke=[]
        self.show_multiple=False

    def explore_hdf_path(self,path):
        total=len(path)
        i=0
        startnode=self.h5f
        while(i<total):
            key=path[i]
            startnode=startnode.get(key)
            i=i+1
        return startnode,key


    def browse_hdf(self):
        self.tre_filelist.clear()
        self.tab_dataset_values.clear()
        self.display_canvas.clear()
        self.currentfilename = str(QFileDialog.getOpenFileName())
        self.txt_filename.setText(self.currentfilename)
        self.h5f=h5py.File(self.currentfilename)
        explore_hdf_n_populate_tree(self.currentfilename, self.tre_filelist)

    def tre_filelist_onclick(self):
        item = self.tre_filelist.selectedItems()
        #print()
        for i in item:
            parent=i.parent()
        current_key=str(item[0].text(0))
        if(parent is not None):
            self.lbl_current_key.setText("Parent Node:"+parent.text(0))
        else:
            self.lbl_current_key.setText("Top level")
        trace_tree_path(item[0],self.current_path)
        self.current_path.append(current_key)
        path=""
        for n in self.current_path:
            path=path+"/"+str(n)
        self.lbl_current_sample.setText(path)
        node,key=self.explore_hdf_path(self.current_path)
        if(type(node)==h5py.Group):
            self.find_group_attrs(node)
        elif(type(node)==h5py.Dataset):
            print("key is ",key)
            self.current_data=np.asarray(node)
            self.find_dataset_values()
        del self.current_path[:]


    def find_group_attrs(self,group):
        self.lst_filecontent.clear()
        attrs_keys=group.attrs.keys()
        total = len(attrs_keys)
        if(total>2):
            self.current_h5_sample=group
        self.lst_filecontent.setRowCount(total)
        self.lst_filecontent.setColumnCount(2)
        for t in range(total):
            header = self.lst_filecontent.horizontalHeader()
            try:
                k = attrs_keys[t].decode("utf-8")
                self.lst_filecontent.setItem(t, 0, QTableWidgetItem(k))
                header.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
                val = group.attrs[k]
                if (type(val) is np.ndarray):
                    shape = len(val)
                    str_out = ""
                    for s in range(shape):
                        str_out = str_out + " " + str(val[s])
                    self.lst_filecontent.setItem(t, 1, QTableWidgetItem(str_out))
                elif (type(val) is str):
                    val = val.decode("utf-8")
                    self.lst_filecontent.setItem(t, 1, QTableWidgetItem(val))
                else:
                    self.lst_filecontent.setItem(t, 1, QTableWidgetItem(str(val)))
                header.setResizeMode(1, QtGui.QHeaderView.ResizeToContents)

                print(k, val)
            except:
                pass
            # self.lst_filecontent.setItem(t, 1, QTableWidgetItem(self.attrs[t][1]))
        self.lst_filecontent.show()


    def find_dataset_values(self):
        shape=self.current_data.shape
        self.tab_dataset_values.clear()
        if(len(shape)==2):
            rows=shape[0]
            cols=shape[1]
            self.tab_dataset_values.setRowCount(rows)
            self.tab_dataset_values.setColumnCount(cols)
            for r in range(rows):
                for c in range(cols):
                    self.tab_dataset_values.setItem(r,c,QTableWidgetItem(str(self.current_data[r][c])))
        elif(len(shape)==1):
            self.tab_dataset_values.setRowCount(shape[0])
            self.tab_dataset_values.setColumnCount(1)
            for r in range(shape[0]):
                self.tab_dataset_values.setItem(r,0,QTableWidgetItem(str(self.current_data[r])))
        self.tab_dataset_values.show()

    def display_multiple_stroke(self):
        if(self.show_multiple):
            for s in self.current_stroke:
                self.show_stroke(s,100)
        else:
            self.show_stroke(self.current_data,100)
        try:
            del self.current_stroke[:]
            del self.current_data[:]
        except:
            pass
        self.show_multiple=False

    def append_stroke(self):
        if not self.show_multiple:
            self.show_multiple=True
        self.current_stroke.append(self.current_data)

    def show_stroke(self,data,new_height):
        total=len(data)
        xs=[]
        ys=[]
        for t in range(total):
            xs.append(data[t][0])
            ys.append(data[t][1])
        try:
            box=self.current_h5_sample.attrs["Box"]
            maxx=int(box[0])
            minx=int(box[1])
            maxy=int(box[2])
            miny=int(box[3])
            print("Global Box found")
        except:
            maxx=max(xs)
            minx=min(xs)
            maxy=max(ys)
            miny=min(ys)
        original_width=maxx-minx
        original_height=maxy-miny
        ar=original_width/float(original_height)
        new_width=ar*new_height

        for t in range(total):
            xn=int(((xs[t]-minx)/float(original_width))*new_width)
            yn=int((ys[t]-miny)/float(original_height)*new_height)
            #print(xn,yn)
            self.display_canvas.addEllipse(xn,yn,3,3,brush=self.colorpallete[self.current_color_index])
        self.current_color_index=(self.current_color_index+1)%len(self.colorpallete)

    def refresh_canvas(self):
        self.display_canvas.clear()



if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    resolution = app.desktop().screenGeometry()
    width,height=resolution.width(),resolution.height()
    print(width,height)
    window = Window([0,0,width,height])
    window.show()
    sys.exit(app.exec_())