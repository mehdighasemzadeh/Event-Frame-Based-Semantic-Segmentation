
#================imports ================
#========= imports libs ===================
import os
import numpy as np
import cv2
import keras 
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
import random
import segmentation_models as sm
from keras import backend as K
import tensorflow_addons as tfa

#============ import custom libs ===========
from model import SegModel



current_dir = os.getcwd()
Image_path = current_dir + "/Dataset/content/our_dataset/test/image/"   
Event_path = current_dir + "/Dataset/content/our_dataset/test/event/"
Label_path = current_dir + "/Dataset/content/our_dataset/test/label/"



loss1 = sm.losses.DiceLoss()
loss2 = sm.losses.CategoricalCELoss()
total_loss = loss1  + loss2

metrics = ['accuracy', sm.metrics.IOUScore(per_image = True)]

model = SegModel(9)



#=================== set optimizer ==============
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps = 2202 * 3 ,
    decay_rate=0.9)

opt = keras.optimizers.Adam(learning_rate=lr_schedule)


#============== compile model ===============================
model.compile(optimizer = opt , loss = total_loss , metrics = metrics)

#========== train model ==================
load_weight_path = "training1/cp.ckpt"
model.load_weights(load_weight_path)



def Label_gen(img):
    out = np.zeros((256,512,9),np.uint8)
    
    out[:,:,0] = np.logical_or(img==0, img==3)

    out[:,:,1] = np.logical_or(img==1 , img==11 , img==2)

    out[:,:,2] = (img==4)
    
    out[:,:,3] = (img==7)


    out[:,:,4] = (img==10)
    
    out[:,:,5] = np.logical_or(img==5,img==12 )

    
    out[:,:,6] = (img==9)
    
    out[:,:,7] = (img==8)
    
    out[:,:,8] = (img==6) #RoadLines
    
    return out




image_ext = "image.png"
event_ext = "events.npy"
label_ext = "gt_labelIds.png"


def EventToImage(input_event , output) :
    temp = input_event[:,:,0:1]
    temp = temp*254
    temp1 = temp.astype(np.uint8)
    output[0:256,0:512,0:1] = temp1
    output[0:256,0:512,1:2] = temp1
    output[0:256,0:512,3:2] = temp1
    
    temp = input_event[:,:,1:2]
    temp = temp*255
    temp1 = temp.astype(np.uint8)
    output[256:512,0:512,0:1] = temp1
    output[256:512,0:512,1:2] = temp1
    output[256:512,0:512,3:2] = temp1
    
    return output


   
def ImageToImage(input_image , output) :
    output[0:256 , 512:1024 , :] = input_image
    return output

   
def BlurToImage(input_image , output) :
    output[256:512 , 512:1024 , :] = input_image
    return output



def PrdictToImage(label , output):
    # ===== label = 0 ============
    temp = np.zeros((256,512,3) , np.uint8)
    for i in range(0,256):
        for j in range(0,512):
            if label[i][j] == 1 :
                temp[i:i+1 , j : j +1 , :]  = [70,70,70] 

            if label[i][j] == 2 :
                temp[i:i+1 , j : j +1 , :]  = [60,20,220]
                
            if label[i][j] == 3 :
                temp[i:i+1 , j : j +1 , :]  = [128,64,128]

            if label[i][j] == 4 :
                temp[i:i+1 , j : j +1 , :]  = [142,0,0]

            if label[i][j] == 5 :
                temp[i:i+1 , j : j +1 , :]  = [153,153,153]

            if label[i][j] == 6 :
                temp[i:i+1 , j : j +1 , :]  = [35,142,107]
                
            if label[i][j] == 7 :
                temp[i:i+1 , j : j +1 , :]  = [232,35,244] 
                
            if label[i][j] == 8 :
                temp[i:i+1 , j : j +1 , :]  = [50,234,157]
                    
                
                
    output[0:256,1024:,:] = temp
    
    return output
                
 


def GTToImage(label , output):
    # ===== label = 0 ============
    temp = np.zeros((256,512,3) , np.uint8)
    for i in range(0,256):
        for j in range(0,512):
            if label[i][j] == 1 :
                temp[i:i+1 , j : j +1 , :]  = [70,70,70] 

            if label[i][j] == 2 :
                temp[i:i+1 , j : j +1 , :]  = [60,20,220]
                
            if label[i][j] == 3 :
                temp[i:i+1 , j : j +1 , :]  = [128,64,128]

            if label[i][j] == 4 :
                temp[i:i+1 , j : j +1 , :]  = [142,0,0]

            if label[i][j] == 5 :
                temp[i:i+1 , j : j +1 , :]  = [153,153,153]

            if label[i][j] == 6 :
                temp[i:i+1 , j : j +1 , :]  = [35,142,107]
                
            if label[i][j] == 7 :
                temp[i:i+1 , j : j +1 , :]  = [232,35,244] 
                
            if label[i][j] == 8 :
                temp[i:i+1 , j : j +1 , :]  = [50,234,157]
                        
                    
          
                
                
    output[256:512,1024:,:] = temp
    
    return output

                
                

########################################################
########################################################
def bluring_img (img,k):
    out = cv2.GaussianBlur(img, (k,k),0) 
    return out

#============= for Single sample =======================   
test_out = np.zeros((512,1536,3) , np.uint8)
name = "05_001_0001_"
for i in range(0,1):
    #======= read event =====================
    input_event = np.load(Event_path+name+event_ext)
    test_out = EventToImage(input_event , test_out)
    event = np.reshape(input_event ,(1,256,512,8))
    #============== read image and scale it ===============
    input_image = cv2.imread(Image_path+name+image_ext,cv2.IMREAD_COLOR)
    test_out = ImageToImage(input_image , test_out) 
    input_image = bluring_img(input_image,1)
    test_out = BlurToImage(input_image , test_out) 
    out = np.zeros((256,512,3) , np.float32)
    for j in range(0,3):
        temp = input_image[:,:,j:j+1]
        max_i = np.amax(temp)
        min_i = np.amin(temp)
        temp1 = (temp - min_i)
        if max_i - min_i > 0 :
            temp1 = temp1/(max_i - min_i)
            temp2 = temp1.astype(np.float32)
            out[:,:,j:j+1] = temp2
    
     
    image = np.reshape(out , (1,256,512,3) )
    
    #event = np.zeros((1,256,512,8) , np.float32)
    test = model.predict([image,event])
    
    
    
    out = np.argmax(test, axis=3)
    out1 = out.reshape(256,512)
    test_out = PrdictToImage(out1,test_out)


    
    l = cv2.imread(Label_path+name+label_ext, 0)
    l1 = Label_gen(l)
    l2 = np.argmax(l1, axis=2)
    l3 = l2.reshape(256,512)
    test_out = GTToImage(l3,test_out)


cv2.imwrite(current_dir +  "/output/test.png", test_out)

plt.imshow(test_out, interpolation='nearest')
plt.show()  





