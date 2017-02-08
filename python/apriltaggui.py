
import apriltag
from apriltag import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('seaborn-paper')


import cv2
have_cv2 = True
# Require cv2, do not even try PIL


gax = [None, None]

def init_detect(familyname='tag36h10'):
    import argparse

    options = argparse.Namespace(border=2, families='[{}]'.format(familyname), nthreads=4, quad_contours=True, quad_decimate=1.0, quad_sigma=0.0, refine_decode=True, refine_edges=False, refine_pose=True, debug=-1)

    print("init_detect({})",familyname)
    det = Detector(options)
    
    return det
def init_gui():
    gax[1] = plt.figure(1,figsize=(12, 8))
    plt.figure(1)
    gax[0] = plt.subplot(111);
    plt.tight_layout()
    
import json    

def toJSON(detections):
    
    data = {'family': detections[0].tag_family.decode(),
            'id': detections[0].tag_id,
            'hamming': detections[0].hamming,
            'goodness': detections[0].goodness,
            'decision_margin': detections[0].decision_margin,
            'H': detections[0].H
           }
    
    print(json.dumps(data));
#                 tag.family.contents.name,
#                 tag.id,
#                 tag.hamming,
#                 tag.goodness,
#                 tag.decision_margin,
#                 homography,
#                 center,
#                 corners

from timeit import default_timer as timer

def show_detect(det, orig, invert=0):

    if len(orig.shape) == 3:
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    else:
        gray = orig
    if (invert):
        gray = 255-gray

    start = timer()
    #for _ in range(100):
    detections, dimg = det.detect(gray, return_image=True)
    end = timer()
    print("Time = ",end - start) 
    
    #toJSON(detections)

    num_detections = len(detections)
    print('Detected {} tags.\n'.format(num_detections))

    for i, detection in enumerate(detections):
        print('Detection {} of {}:'.format(i+1, num_detections))
        print()
        print(detection.tostring(indent=2))
        print()

    plt.sca(gax[0])
    plt.cla()
    plt.gray()
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.draw()
    
    for D in detections:
        print(repr(D))
        c=D.corners
        print(c[:,1])
        pp,=plt.plot(c[:,0],c[:,1],'-')
        plt.plot(c[:,0],c[:,1],'o',markeredgecolor=pp.get_color(), markerfacecolor="None")
        plt.text(np.mean(c[:,0]),np.min(c[:,1])-5,str(D.tag_id),fontsize=12,horizontalalignment='center', verticalalignment='bottom', color=pp.get_color())
        
        if (D.tag_id>=0):
            H = D.homography;
            
            x=0; y=0;
            
            #sv.libc.homography_project(H, 0, 0, x, y);
    plt.autoscale(enable=True, tight=True)
    plt.draw()
    
    plt.show(block=False)
    
    
def tagradius(D):
    x=[D.corners[i,0] for i in range(4)]
    y=[D.corners[i,1] for i in range(4)]
    return max([max(x)-min(x), max(y)-min(y)])
    
def draw_detections(tagimg, detections, draw_sampling=0, textthickness=2, fontscale=1.0):
    
    for D in detections:
        #print(repr(D))
        c=D.corners
        
        def homography_project(H, x,y):
            u = H[0,0]*x+H[0,1]*y+H[0,2]
            v = H[1,0]*x+H[1,1]*y+H[1,2]
            w = H[2,0]*x+H[2,1]*y+H[2,2]
        
            return u/w,v/w
        
        cv2.polylines(tagimg, np.int32([c.reshape(-1, 1,2)]), False, (255,0,0))
        
        st=str(D.tag_id)
        sz,baseline=cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX,fontscale,textthickness)
        cv2.putText(tagimg, st, tuple(np.int32([np.mean(c[:,0])-sz[0]/2,np.min(c[:,1])-5])),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale,color=(255,0,0),thickness=textthickness)
        
        cv2.putText(tagimg, "{}px".format(int(tagradius(D))), tuple(np.int32([np.mean(c[:,0])-sz[0]/2,np.max(c[:,1])+5])),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale/2,color=(0,0,255),thickness=1)
        
        if (D.tag_id>=0):
            H = D.homography;
            
            x=0; y=0;
            
            #sv.libc.homography_project(H, 0, 0, x, y);
            
        if (draw_sampling==1):
            H = D.homography;
            
            for x in np.array([-2,-1,0,1,2])/7*2:
                for y in np.array([-2,-1,0,1,2])/7*2:
                    x2,y2 = homography_project(H, x,y);
                    x2=int(x2); y2=int(y2);
                    #print("x2={}, y2={}".format(x2,y2));
                    cv2.line(tagimg, (x2,y2), (x2,y2), (0,255,0),1)
                    
        if (draw_sampling==2):
            H = D.homography;
            
            for x in np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5])/7*2:
                x1,y1 = homography_project(H, x,-2.5/7*2);
                x2,y2 = homography_project(H, x,+2.5/7*2);
                x2=int(x2); y2=int(y2);                x1=int(x1); y1=int(y1);
                cv2.line(tagimg, (x1,y1), (x2,y2), (0,255,0),1)
                
                x1,y1 = homography_project(H, -2.5/7*2,x);
                x2,y2 = homography_project(H, +2.5/7*2,x);
                x2=int(x2); y2=int(y2);                x1=int(x1); y1=int(y1);
                cv2.line(tagimg, (x1,y1), (x2,y2), (0,255,0),1)
    
def draw_detect(det, orig, invert=0, draw_sampling=0, thickness=3, fontscale=1.0): # => tagimg

    if len(orig.shape) == 3:
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    else:
        gray = orig
    if (invert):
        gray = 255-gray

    detections = det.detect(gray, return_image=False)

    num_detections = len(detections)
    print('Detected {} tags.\n'.format(num_detections))

#     for i, detection in enumerate(detections):
#         print('Detection {} of {}:'.format(i+1, num_detections))
#         print()
#         print(detection.tostring(indent=2))
#         print()

    tagimg = orig.copy()
    
    draw_detections(tagimg, detections, draw_sampling, textthickness=thickness, fontscale=fontscale)
            
    return tagimg

def save_fig():
        pp = PdfPages('foo.pdf')
        pp.savefig(gax[1])
        pp.close()

def main():

    '''Test function for this Python wrapper.'''

    from argparse import ArgumentParser

    # for some reason pylint complains about members being undefined :(
    # pylint: disable=E1101

    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('filenames', metavar='IMAGE', nargs='*',
                        help='files to convert')

    parser.add_argument('-V', dest='video_in', default=None,
                        help='Input video')
    parser.add_argument('-f0', dest='f0', default=0, type=int,
                        help='Frame start')
    parser.add_argument('-f1', dest='f1', default=0, type=int,
                        help='Frame end')
    parser.add_argument('-I', dest='inverse', default=False, 
                        action='store_true',
                        help='Process inverse image')
                        
    apriltag.add_arguments(parser)

    options = parser.parse_args()
    
    print(options)
    
    #det = init_detect(options)
    det = apriltag.Detector(options)
    
    #det.tag_detector.contents.qcp.min_side_length = 25;
    #det.tag_detector.contents.qcp.min_aspect_ratio = 0.7;
    #x = det.tag_detector.tata
    
    print(options)

    if (options.video_in): # Input is a video
        print('Processing video {}'.format(options.video_in))
        vidcap = cv2.VideoCapture(options.video_in)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,0)     
        nframes=vidcap.get(cv2.CAP_PROP_FRAME_COUNT);
        options.f1=int(min(options.f1,nframes))
        
        win = cv2.namedWindow('tags',cv2.WINDOW_NORMAL)
        
        for f in range(options.f0,options.f1+1):
        
            filename="tagout/tagout_{:05d}.png".format(f)
                        
            fps=22
            vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*f)    
            status,orig = vidcap.read();
            if (orig is None):
                print('Warning: could not read frame {}'.format(f))
                continue
            print("Detecting on frame {}, saving to {}".format(f,filename))
            #orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB);
            gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
            if (options.inverse):
                gray = 255-gray
            
            tagimg = draw_detect(det, orig, draw_sampling=0)
            #tagimg = cv2.cvtColor(tagimg, cv2.COLOR_RGB2BGR);
                            
            show_detect(det, orig)
                            
            #cv2.imshow('tags', cv2.resize(tagimg,(0,0),fx=0.5,fy=0.5))
            #cv2.waitKey(0)
            
            cv2.imwrite(filename,tagimg)
            
            plt.show(block=True)
    else: # Input is an image (or several)
        print('Processing image(s) {}'.format(options.filenames))
        for filename in options.filenames:

            if have_cv2:
                orig = cv2.imread(filename)
                if len(orig.shape) == 3:
                    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
                else:
                    gray = orig
                #cv2.imshow('win', gray)
                #while cv2.waitKey(5) < 0:
                #    pass
            else:
                pil_image = Image.open(filename)
                orig = numpy.array(pil_image)
                gray = numpy.array(pil_image.convert('L'))
            if (options.inverse):
                gray = 255-gray

            detections = det.detect(gray, return_image=False)
            #detections, dimg = det.detect(gray, return_image=True)

            num_detections = len(detections)
            print('Detected {} tags.\n'.format(num_detections))

            for i, detection in enumerate(detections):
                print('Detection {} of {}:'.format(i+1, num_detections))
                print()
                print(detection.tostring(indent=2))
                print()

            plt.gray()
            plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        
            for D in detections:
                print(repr(D))
                c=D.corners
                print(c[:,1])
                pp,=plt.plot(c[:,0],c[:,1],'-')
                plt.plot(c[:,0],c[:,1],'o',markeredgecolor=pp.get_color(), markerfacecolor="None")
                plt.text(np.mean(c[:,0]),np.min(c[:,1])-5,str(D.tag_id),fontsize=12,horizontalalignment='center', verticalalignment='bottom', color=pp.get_color())
                plt.tight_layout()
                
                h,w, = gray.shape
                plt.gca().set_xlim(left=0, right=w)
                plt.gca().set_ylim(top=0, bottom=h)
        
            plt.show(block=True)

#             if len(orig.shape) == 3:
#                 overlay = orig / 2 + dimg[:, :, None] / 2
#             else:
#                 overlay = gray / 2 + dimg / 2
# 
#             if have_cv2:
#                 overlay=cv2.resize(overlay, (0,0), fx=0.25, fy=0.25)
#                 cv2.imshow('win', overlay/255)
#                 while cv2.waitKey(5) < 0:
#                     pass
#             else:
#                 output = Image.fromarray(overlay)
#                 output.save('detections.png')

if __name__ == '__main__':

    main()
