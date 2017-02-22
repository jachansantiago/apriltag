#!/usr/bin/env python

"""Additional wrappers around apriltag to interact with OpenCV and matplotlib.

Author: Rémi Mégret, 2017
"""


import apriltag
#from apriltag import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os
import math

plt.style.use('seaborn-paper')


import cv2
have_cv2 = True
# Require cv2, do not even try PIL


gax = [None, None]

import argparse

def init_detection(config='tag36h10'):

    presets = {
        'tag36h10': argparse.Namespace(
            border=2, families='[{}]'.format('tag36h10'), nthreads=4, quad_contours=True, quad_decimate=1.0, quad_sigma=0.0, refine_decode=True, refine_edges=False, refine_pose=True, debug=-1, inverse=False),
        'tagbeetag': argparse.Namespace(
            border=2, families='[{}]'.format('tagbeetag'), nthreads=4, quad_contours=True, quad_decimate=1.0, quad_sigma=0.0, refine_decode=True, refine_edges=False, refine_pose=True, debug=-1, inverse=True)
    }
    
    options = presets[config]
        
    print("init_detection({})",config)
    print("Options=",options)
    det = apriltag.Detector(options)
    
    options.min_side_length = 15
    options.min_aspect = 0.7
    
    #det.tag_detector.contents.qcp.min_side_length = options.min_side_length;
    #det.tag_detector.contents.qcp.min_aspect = options.min_aspect;
    
    return det

# def init_detect(familyname='tag36h10'):
# 
#     options = argparse.Namespace(border=2, families='[{}]'.format(familyname), nthreads=4, quad_contours=True, quad_decimate=1.0, quad_sigma=0.0, refine_decode=True, refine_edges=False, refine_pose=True, debug=-1, inverse=False)
# 
#     print("init_detect({})",familyname)
#     det = apriltag.Detector(options)
#     
#     return det
    
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

from timeit import default_timer as timer

def do_detect(det, orig):
    if len(orig.shape) == 3:
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    else:
        gray = orig
    # Inversion done directly in Detector now
    #if (invert):
    #    gray = 255-gray

    start = timer()
    detections = det.detect(gray, return_image=False)
    end = timer()
    #print("Time = ",end - start) 
    
    return detections

def print_detections(detections, show_details=False):
    num_detections = len(detections)
    print('Detected {} tags.\n'.format(num_detections))

    if (show_details):
        for i, detection in enumerate(detections):
            print('Detection {} of {}:'.format(i+1, num_detections))
            print()
            print(detection.tostring(indent=2))
            print()    

def plot_detections(detections, ax, orig):
    """Plots apriltag detections on matplotlib axis"""

    plt.sca(ax)
    plt.cla()
    plt.gray()
    if len(orig.shape) == 3:
        bgimg = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    else:
        bgimg = orig.copy()
    plt.imshow(bgimg)
    #plt.draw()
    
    for D in detections:
        #print(repr(D))
        c=D.corners
        #print(c[:,1])
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


# obsolete
def show_detect(det, orig):
    """Warning: obsolete show_detect"""

    detections = do_detect(det, orig)
    
    #toJSON(detections)

    #print_detections(detections)

    plot_detections(detections, gax[0], orig)
    
    
def tagradius(D):
    x=[D.corners[i,0] for i in range(4)]
    y=[D.corners[i,1] for i in range(4)]
    return max([max(x)-min(x), max(y)-min(y)])
    
def draw_detections(tagimg, detections, draw_sampling=0, 
                    textthickness=2, fontscale=1.0):
    
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
    
def draw_detect(det, orig, draw_sampling=0, 
                thickness=3, fontscale=1.0): # => tagimg

    detections = do_detect(det, orig)

    #print_detections(detections, show_details=False)

    tagimg = orig.copy()
    
    draw_detections(tagimg, detections, draw_sampling, 
                    textthickness=thickness, fontscale=fontscale)
            
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
        
    show_default = ' (default %(default)s)'

    parser.add_argument('filenames', metavar='IMAGE', nargs='*',
                        help='files to convert')

    parser.add_argument('-V', dest='video_in', default=None,
                        help='Input video')
    parser.add_argument('-f0', dest='f0', default=0, type=int,
                        help='Frame start '+ show_default)
    parser.add_argument('-f1', dest='f1', default=0, type=int,
                        help='Frame end '+ show_default)
                        
    parser.add_argument('-min_side_length', dest='min_side_length', default=25, 
                        type=int,
                        help='Tags smaller than that are discarded '+ show_default)
    parser.add_argument('-min_aspect', dest='min_aspect',
                        default=0.7, type=float,
                        help='Tags smaller than that are discarded '+ show_default)
                        
    parser.add_argument('-nowait', dest='nowait', default=False, 
                        action='store_true',
                        help='Process all input without waiting for user')
                        
    apriltag.add_arguments(parser)

    options = parser.parse_args()
    
    if (options.f1<options.f0):
        options.f1=options.f0
    
    print(options)
    
    det = apriltag.Detector(options)
    
    # For Quad Contour Params Detection (QCP)
    #det.tag_detector.contents.qcp.min_side_length = options.min_side_length;
    #det.tag_detector.contents.qcp.min_aspect = options.min_aspect;
    
# Defaults for Quad Threshold Approach
#     qtp->max_nmaxima = 10;
#     qtp->min_cluster_pixels = 5;
#     qtp->max_line_fit_mse = 1.0;
#     qtp->critical_rad = 10 * M_PI / 180;
#     qtp->deglitch = 0;
#     qtp->min_white_black_diff = 15;
    
    qtp = det.tag_detector.contents.qtp
    qtp.min_cluster_pixels = 200
    qtp.critical_rad = 50.0 / 180 * math.pi
    qtp.min_white_black_diff = 30
    qtp.deglitch = 0

#         ('max_nmaxima', ctypes.c_int),
#        ('max_line_fit_mse', ctypes.c_float),
    
    init_gui()

    if (options.video_in): # Input is a video
        print('Processing video {}'.format(options.video_in))
        vidcap = cv2.VideoCapture(options.video_in)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,0)     
        nframes=vidcap.get(cv2.CAP_PROP_FRAME_COUNT);
        options.f1=int(min(options.f1,nframes))
        
        win = cv2.namedWindow('tags',cv2.WINDOW_NORMAL)
        
        os.makedirs('tagout',exist_ok=True)
        
        for f in range(options.f0,options.f1+1):
        
            filename="tagout/tagout_{:05d}.png".format(f)
                        
            fps=22
            vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*f)    
            status,orig = vidcap.read();
            # Caution: orig in BGR format by default
            if (orig is None):
                print('Warning: could not read frame {}'.format(f))
                continue
            print("Detecting on frame {}, saving to {}".format(f,filename))
            
            detections = do_detect(det, orig)

            plot_detections(detections, gax[0], orig)
            
            tagimg = orig.copy()
            draw_detections(tagimg, detections, draw_sampling=0)            
            #tagimg = cv2.cvtColor(tagimg, cv2.COLOR_RGB2BGR);
            cv2.imwrite(filename,tagimg)
            
            #print_detections(detections, show_details=False)
            
            plt.draw()
            plt.show(block=False)
            
            #_ = input("Stopped at frame {}. Press Enter to continue".format(f))
    else: # Input is an image (or several)
        print('Processing image(s) {}'.format(options.filenames))
        for filename in options.filenames:

            if have_cv2:
                orig = cv2.imread(filename)
            else:
                pil_image = Image.open(filename)
                orig = numpy.array(pil_image)
                #gray = numpy.array(pil_image.convert('L'))
                
            detections = do_detect(det, orig)
            
            plot_detections(detections, gax[0], orig)
            
            tagimg = orig.copy()
            draw_detections(tagimg, detections, draw_sampling=0)            
            #tagimg = cv2.cvtColor(tagimg, cv2.COLOR_RGB2BGR);
            cv2.imwrite(filename,tagimg)
            
            plt.show(block=False)

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
