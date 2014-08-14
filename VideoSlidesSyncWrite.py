
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import cv
import sys


# In[2]:

from IPython.html.widgets import interact, interactive, fixed
from IPython.display import clear_output, display, HTML
from IPython.html import widgets


# In[3]:

from io import BytesIO
import PIL
from IPython.display import display, Image

def display_img_array(ima, cvt=None, **kwargs):
    if cvt:
        ima = cv2.cvtColor(ima, cvt)
    im = PIL.Image.fromarray(ima)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png', **kwargs))


# In[4]:

def normalize(im):    
    im=cv2.cvtColor(im, cv2.CV_32F)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im=cv2.equalizeHist(im)
    return im


# In[5]:

def gray2vector(img):
    v = img.reshape(-1).astype(float) 
    v = v - np.average(v) 
    return v/np.linalg.norm(v)


# In[6]:

def diff_i(gray, i):
    v1 = gray2vector(gray)
    return np.dot(v1, slides_v[i])

def compare_slides(gray, slides_v):
    v1 = gray2vector(gray)
    r = np.dot(slides_v, v1)
    i = np.argmax(r)
    return r[i], i

def compare_absdiff(gray):
    return sorted( (cv2.absdiff(slides[i], gray).sum() , i)  for i in range(len(slides)))


# In[7]:

def frame_to_time(f, r):
    s = f/r
    m = int(s/60)
    s = s -60*m
    return "%d:%04.1f"%(m,s)
def sync_video( fn, p, slides, slides_v, threshold=0.8, step=10, dark=1000, STOP=-1, debug=False):
    cap = cv2.VideoCapture(fn)
    frame_rate = cap.get(cv.CV_CAP_PROP_FPS)
    print "frame_rate", frame_rate
    num_of_frames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    frame_index =-1
    last_slide = -1
    last_start = -1
    frame_list = []
    progress = widgets.IntProgressWidget(min=0, max = num_of_frames - 1, value=0)
    progress_text = widgets.TextWidget()
    progress.set_css('background', 'black')
    display(progress)
    display(progress_text)
    while cap.isOpened():
        frame_index +=1
        ret, frame = cap.read()
        if not ret:
            break        
        if frame_index%step ==0:
            if STOP!=-1 and frame_index > STOP:
                break                        
            gray = cv2.resize(normalize(frame)[p[0]:p[2],p[1]:p[3]], (256,256))        
            darklevel = np.linalg.norm(gray.reshape(-1).astype(float)) 
            if darklevel < dark:
                # too dark                
                this_slide, v, i = -1, -1, 0
            else:            
                v, i = compare_slides(gray, slides_v)
                this_slide = i if v > threshold else -1 
            if debug:
                if i>=0:
                    frame2 = frame.copy()            
                    frame2[p[0]:p[2], p[1]:p[3]] = cv2.resize(original_slides[i][q[0]:q[2], q[1]:q[3]], (p[3]-p[1], p[2]-p[0]))
                    outp = np.concatenate( (frame2, cv2.addWeighted(frame,0.5,frame2, 0.5,0), frame), axis = 1)
                    display_img_array(outp, width=1200, cvt=cv2.COLOR_BGR2RGB)
                else:
                    display_img_array(frame, width=400, cvt=cv2.COLOR_BGR2RGB)
                print v,i
            if frame_index%100 ==0:
                progress.value = frame_index            
                progress_text.value = "%d/%d (%.1f)"%(frame_index, num_of_frames, 100.0*frame_index/num_of_frames)
            if this_slide != last_slide:
                # update
                frame_list.append( (last_start,  frame_index-1, last_slide))
                
                # display information                
                if last_slide >=0:                
                    fl = frame_list[-1]
                    t1, t2 = frame_to_time(fl[0], frame_rate), frame_to_time(fl[1], frame_rate)
                    print fl, "=(%s, %s)"%(t1,t2), "v=%f"%v, "dark=%d"%darklevel
                last_start = frame_index                
                if debug:
                    p1 = cv2.resize(frame, (320,240))
                    if i>=0:
                        p2 = cv2.resize(original_slides[i], (320,240))
                        p1 = np.concatenate( (p1, p2), axis =1)                
                    if last_slide >= 0:
                        display_img_array(cv2.cvtColor(p1, cv2.COLOR_BGR2RGB) , width=200)                 
        last_slide = this_slide
    # last update
    frame_list.append( (last_start,  frame_index-1, last_slide))
    cap.release()
    return frame_list


# In[8]:

def write_file(fn, p,q, outfn, original_slides, sync_result, M=20, fourcc="XVID", SKIP=None):
    W,H = 1440, 960
    W,H = 1920, 1080

    cap = cv2.VideoCapture(fn)
    SW, SH = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    p2 = ( p[0]*H/SH, p[1]*W/SW, p[2]*H/SH, p[3]*W/SW)
    pw, ph = p2[3]-p2[1], p2[2]-p2[0]
    print p2, q
    fourcc = cv.FOURCC(*fourcc)
    num_of_frames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    frame_rate = cap.get(cv.CV_CAP_PROP_FPS)
    print "frame_rate", frame_rate
    sys.stdout.flush()
    out = cv2.VideoWriter(outfn, fourcc, frame_rate, (W, H))
    frame_index =-1
    last_slide = -1
    last_start = -1
    frame_list = []
    result_index = 0
    progress = widgets.IntProgressWidget(min=0, max = num_of_frames - 1, value=0)
    progress_text = widgets.TextWidget()
    progress.set_css('background', 'black')
    display(progress)
    display(progress_text)
    while cap.isOpened():
        frame_index +=1
        ret, frame = cap.read()
        if not ret:
            break
        while result_index < len(sync_result) and sync_result[result_index][1] < frame_index:
            result_index += 1            
        the_slide = (-1,-1,-1) if result_index >= len(sync_result) else sync_result[result_index]
        if SKIP and the_slide[2] in SKIP:
            the_slide = (-1,-1,-1)
        original_frame = cv2.resize(frame, (W, H), interpolation = cv2.INTER_CUBIC)
        if the_slide[2] >=0 and the_slide[1]-the_slide[0]>3*M:
            slide = original_slides[the_slide[2]]
            inner_frame = cv2.resize(slide[q[0]:q[2], q[1]:q[3]],  (pw, ph), interpolation = cv2.INTER_CUBIC )
            d = min(frame_index-the_slide[0], the_slide[1]-frame_index)
            out_frame = original_frame.copy()
            out_frame[p2[0]:p2[2], p2[1]:p2[3]] = inner_frame
            if d < M:
                out_frame = cv2.addWeighted(out_frame, d*1.0/M , original_frame, 1- d*1.0/M, 0)
        else:
            out_frame = original_frame
        out.write(out_frame)
        if frame_index%100 ==0:
            progress.value = frame_index            
            progress_text.value = "%d/%d (%.1f)"%(frame_index, num_of_frames, 100.0*frame_index/num_of_frames)
            if frame_index%10000 ==0:
                disp_frame = np.concatenate((out_frame[:, :W/2], original_frame[:,W/2:]), axis=1)            
                display_img_array(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB) , width=200) 
    cap.release()
    out.release()
            


# In[9]:

import os.path
def load_original_slides(name):
    original_slides = []
    i = 0
    #progress = widgets.IntProgressWidget(min=0, max = num_of_frames - 1, value=0)
    progress_text = widgets.TextWidget()
    #progress.set_css('background', 'black')
    #display(progress)
    display(progress_text)
    while True:         
        progress_text.value = "loading %d"%i
        img = cv2.imread("%s/%s-%d.png"%(name, name, i))
        if img is None:
            break        
        original_slides.append(img)
        i+=1
    print "load original slides", len(original_slides)
    return original_slides
def prepare_slides(original_slides, q, blur_factor):
    normalized_slides = (cv2.blur(normalize(s), (blur_factor, blur_factor))  for s in original_slides)
    slides = [cv2.resize(s[q[0]:q[2], q[1]:q[3]], (256,256), interpolation = cv2.INTER_CUBIC)  for s in normalized_slides]
    slides_v = np.array([gray2vector(s) for s in slides])
    print "slides prepared"
    return slides, slides_v
    


# In[10]:

original_slides, original_slides_name, result = None, None, None


# In[11]:

def auto_sync(NAME, p1, q1, blur_factor, p2=None, q2=None,  threshold=0.8, step=10, dark=1500, STOP=-1, debug=False, 
              SKIP=None, M=20, PASS=[2,3],  fourcc="XVID",  EXT="avi"):
    print "NAME=", NAME
    fn_base = "%s/%s"%(NAME,NAME)
    if  os.path.isfile(fn_base+".mp4"):
        fn = fn_base+".mp4"
    elif os.path.isfile(fn_base+".avi"):
        fn = fn_base+".avi"
    else:
        print "original video file does not exist"
        return
    outfn = "%s/better_%s.%s"%(NAME, NAME, EXT)
    global original_slides, result, original_slides_name    
    if 0 in PASS or not os.path.isfile("%s/%s-0.png"%(NAME,NAME)) : # 0 Extract PDF
        print "extract slides"
        sys.stdout.flush()
        print os.system("convert -density 200 %s/%s.pdf  %s/%s.png"%(NAME,NAME,NAME,NAME))
    
    if 1 in PASS or original_slides_name != NAME:
        print "load original png"
        original_slides = load_original_slides(NAME)
        original_slides_name = NAME            

    if 2 in PASS or original_slides_name != NAME: # Sync Video and Slides
        print "prepare slides"
        slides, slides_v = prepare_slides(original_slides, q1, blur_factor)        
        print "syncing video"
        result = sync_video(fn, p1, slides, slides_v, threshold=threshold, step=step, dark=dark, STOP=STOP, debug=debug)
        print "sync_video done"   
        
    if p2 is None:   # full screen
        p2 = (0,0,1080,1920)
    if q2 is None:  # full screen
        q2 = (0,0)+tuple(original_slides[0].shape[:2])
        
    if 3 in PASS or original_slides_name != NAME:
        print "start writing and converting"
        TEMP_OUT = "temp_out."+EXT
        write_file(fn, p1, q1, TEMP_OUT, original_slides, result, M=M, fourcc=fourcc, SKIP=SKIP)
        print "write done"
        sys.stdout.flush()
        retcode = os.system("avconv -y -i %s -i %s -map 0:v -map 1:a -c:v copy -c:a copy %s"%(TEMP_OUT, fn, outfn))
        print "covert done", retcode


# In[12]:

p1,q1, blur_factor = (10, 160, 1080, 1754) , (0, 40, 2112, 2844) , 10 
p2, q2 = p1, q1
auto_sync("tulip",  p1,q1, blur_factor, p2,q2, threshold=0.9, M=5, fourcc="x264", EXT="mp4", SKIP=[28], PASS=[3])


# In[54]:

p1,q1,blur_factor = (10, 159, 1080, 1754) , (0, 39, 2112, 2844) , 16
p2, q2 =  (10, 131, 1080, 1750) , (0, 0, 2115, 2844),
auto_sync("graphtool", p1,q1, blur_factor, p2, q2, threshold=0.8, SKIP=[90, 91,92,93,94,95,96,97], PASS=[3])


# In[ ]:

p1,q1, blur_factor = (10, 159, 1080, 1750) , (0, 35, 2115, 2844), 16
p2,q2 = (10, 138, 1080, 1750) , (0, 0, 2115, 2844) # test abcdefg
auto_sync("ls",  p1,q1, blur_factor, p2,q2, threshold=0.8, SKIP=[4,35,36] )


# In[ ]:

p1,q1, blur_factor = (211, 281, 958, 1723) , (0, 68, 2133, 2838) , 18
p2,q2 = None, None 
auto_sync("fabric",  p1,q1, blur_factor, p2,q2, threshold=0.8, SKIP=[19], M=40)


# In[ ]:

p1,q1, blur_factor = (145, 160, 954, 1751) , (0, 27, 1125, 2000) , 12
p2, q2 = None, None
auto_sync("vote",  p1,q1, blur_factor, p2,q2, threshold=0.75, M=2)


# In[ ]:

p1,q1, blur_factor = (11, 144, 1080, 1752) , (0, 0, 1485, 2000) , 16
p2, q2 = p1, q1
auto_sync("mezz",  p1,q1, blur_factor, p2,q2, threshold=0.9, M=20)


# In[ ]:

p1,q1, blur_factor = (11, 135, 1080, 1753) , (0, 0, 2113, 2844) , 16
p2, q2 = p1, q1
auto_sync("summly",  p1,q1, blur_factor, p2,q2, threshold=0.935, M=20)


# In[ ]:

p1,q1, blur_factor = (19, 160, 1067, 1672) , (0, 80, 2070, 2844), 36
p2, q2 = p1, q1
auto_sync("StreetVoice",  p1,q1, blur_factor, p2,q2, threshold=0.5, M=20)


# In[ ]:

p1,q1, blur_factor = (19, 108, 1080, 1688) , (0, 0, 2084, 2844), 26
p2, q2 = p1, q1
auto_sync("grs",  p1,q1, blur_factor, p2,q2, threshold=0.7, M=20)


# In[48]:

ip.set_css('background', 'black')

