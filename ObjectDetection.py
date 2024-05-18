from ultralytics import YOLO
import cvzone
import math
import cv2
import numpy as np


class ImgPr:
    img=''
    def __init__(self, location):
        self.location=location

    def imageRead(self):
        self.img=cv2.imread(self.location)
    
    def delWin(self):
        cv2.destroyAllWindows()
        
    # Showing image in window
    def showImage(self):
        cv2.imshow('Output', self.img)
        cv2.waitKey(0)    
        
    # Converting into gray color
    def imgGray(self):
        self.imgGray1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray Image', self.imgGray1)
        cv2.waitKey(0)
        
    # Converting into blur image
    def imgBlur(self):
        self.imgBlur1 = cv2.GaussianBlur(self.img, (7,7), 0)
        cv2.imshow('Blur Image', self.imgBlur1)
        cv2.waitKey(0)
        
    # Image edges finding
    def imgEdges(self, a, b):
        self.imgEdges1 = cv2.Canny(self.img, a, b)
        cv2.imshow('Image edges', self.imgEdges1)
        cv2.waitKey(0)
        
    # Image Cropping
    def imgCrop(self, a):
        self.height, self.width,_=self.img.shape
        if a=='up_R':
            self.imgCropped = self.img[0:int(self.height/2), 0:int(self.width/2)]
        elif a=='up_L':
            self.imgCropped = self.img[0:int(self.height/2), int(self.width/2):-1]
        elif a=='dn_R':
            self.imgCropped = self.img[int(self.height/2):-1, 0:int(self.width/2)]
        elif a=='dn_L':
            self.imgCropped = self.img[int(self.height/2):-1, int(self.width/2):-1]
        elif a=='ctr':
            self.imgCropped = self.img[int(self.height/4):int(self.height-self.height/4), int(self.width/4):int(self.width-self.width/4)]
        else:
            print("Not a valid argument!")
        cv2.imshow('Output1', self.imgCropped)
        cv2.waitKey(0)
                
    # Rotating image
    def imgRotate(self, a):
        r, c, _ = self.img.shape
        self.imgTemp = self.img.copy()
        
        # Flip the image vertically
        if a == 'vr':
            for i in range(r // 2):
                temp = self.imgTemp[i].copy()
                self.imgTemp[i] = self.imgTemp[-1 - i]
                self.imgTemp[-1 - i] = temp
        # Flip the image horizontally
        elif a=='hr':
            for i in range(r):
                for j in range(c // 2):
                    self.imgTemp[i][j], self.imgTemp[i][c - 1 - j] = self.imgTemp[i][c - 1 - j].copy(), self.imgTemp[i][j].copy()

        cv2.imshow("Output", self.imgTemp)
        cv2.waitKey(0)
    
    # Image resizing
    def imgResize(self, im):
        self.im=im
        self.rk, self.ck,_ = self.im.shape 
        # print('rk:',self.rk,'ck:',self.ck)
        self.imgResized = cv2.resize(self.im, (self.ck-int(self.ck/1.5), self.rk-int(self.rk/1.5)))
        return self.imgResized
    
    # Image joiner
    def imageJoiner(self, imgArray):
        self.imgArray=imgArray
        rows = len(self.imgArray)
        cols = len(self.imgArray[0])
        print('rows:',rows,'cols:',cols)
        
        self.r, self.c,_ = self.imgArray[0][0].shape
        for i in range(rows):
            for j in range(cols):
                self.r1, self.c1,_ = self.imgArray[i][j].shape
                if self.r > self.r1:
                    self.r=self.r1
                if self.c > self.c1:
                    self.c=self.c1
        print('r=',self.r,'c=',self.c)
        self.arr = []
        for i in range(rows):
            for j in range(cols):
                self.height, self.width,_ = self.imgArray[i][j].shape
                self.imgArray[i][j] = self.imgArray[i][j][int(self.height/2-self.r/2):int(self.height/2+self.r/2), int(self.width/2-self.c/2):int(self.width/2+self.c/2)]
                
            self.arr+=[np.hstack(tuple(self.imgArray[i]))]
            self.arr[i] = self.imgResize(self.arr[i])
            cv2.imshow("Horizontal", self.arr[i])
            cv2.waitKey(0)
                           
        self.imgVer = np.vstack(tuple(self.arr))
        cv2.imshow("Vertical", self.imgVer)
        cv2.waitKey(0)

        

class Object_Detection:
    def __init__(self):
        
        self.a=input('\nWedcam(w) or Saved video(v) or img(i): ')
        if self.a=='w':
            self.cap = cv2.VideoCapture(0)  # For Webcam
            self.cap.set(3, 1280)
            self.cap.set(4, 720)
        elif self.a=='v':
            # self.cap = cv2.VideoCapture("opencv/ppe-1-1.mp4")  # For Video
            # self.cap = cv2.VideoCapture("opencv/ppe-2-1.mp4")  # For Video
            # self.cap = cv2.VideoCapture("opencv/ppe-3-1.mp4")  # For Video
            self.cap = cv2.VideoCapture("opencv/dj.mp4")  # For Video
        else:
            self.im = cv2.imread('opencv/112.png') # For image

        self.b=input("\nFor predict file:\nClone Cancer(c) or Object detect(o): ")
        if self.b=='o':
            self.model = YOLO("opencv/Audity.pt")
            self.classNames = ['Shoes','Watch', 'Mobile', 'ID Card', 'Sunglass', 'Female', 'Male']
        elif self.b=='c':
            self.model = YOLO("opencv/best (7).pt")
            self.classNames = ['Colon_Cancer']
        
        
    def detect(self):   
        while True:
            if self.a=='i':
                img = self.im
            else:
                success, img = self.cap.read()
            
            results = self.model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])

                    cvzone.putTextRect(img, f'{self.classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)


            cv2.imshow("Image", img)
            if self.a=='i':
                cv2.waitKey(0)
                break
            else:
                cv2.waitKey(1)
 
# obd1=Object_Detection()   
# obd1.detect()

class Open_Vision(ImgPr, Object_Detection):
    def __init__(self, location=-1):
        if location!=-1:
            ImgPr.__init__(self, location)
        else:
            Object_Detection.__init__(self)
   
   
   
a=input("Image processing(im) or Object detection(ob): ")
if a=='ob':         
    obd1=Open_Vision()   
    obd1.detect()
else:
    ob1=Open_Vision('opencv/dgfdf.jpeg')
    ob1.imageRead()
    ob1.showImage()
    ob1.imgGray()
    ob1.imgBlur()
    ob1.imgEdges(150, 200)
    ob1.delWin()
    img1 = cv2.imread('opencv/photo_6199231756847136461_y.jpg')
    img2 = cv2.imread('opencv/audi.jpg')
    img3 = cv2.imread('opencv/nigga.jpg')
    img4 = cv2.imread('opencv/441318934_362380056829937_2608713035786839072_n.jpg')
    img5 = cv2.imread('opencv/photo_6199231756847136464_y.jpg')
    img6 = cv2.imread('opencv/441330735_362379963496613_3421901684644553621_n.jpg')
    cv2.imshow("img1", ob1.imgResize(img1))
    cv2.waitKey(0)
    cv2.imshow("img2", ob1.imgResize(img2))
    cv2.waitKey(0)
    cv2.imshow("img3", ob1.imgResize(img3))
    cv2.waitKey(0)
    cv2.imshow("img4", ob1.imgResize(img4))
    cv2.waitKey(0)
    cv2.imshow("img5", ob1.imgResize(img5))
    cv2.waitKey(0)
    cv2.imshow("img6", ob1.imgResize(img6))
    cv2.waitKey(0)
    print('l1:',img1.shape)
    print('l2:',img2.shape)
    ob1.imageJoiner([[img1, img2, img3],[img4,img5,img6]])
    ob1.delWin()

    ob2=Open_Vision('opencv/Lambo.jpg')
    ob2.imageRead()
    ob2.showImage()
    ob2.imgCrop('up_R')
    ob2.imgRotate('hr')