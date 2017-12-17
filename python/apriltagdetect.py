#!/usr/bin/env python

"""Use apriltag with OpenCV to detect in videos.

Author: Rémi Mégret, 2017
"""

use_pympler = False
if (use_pympler):
    from pympler import muppy,summary,tracker
use_resource = True
if (use_resource):
  import resource

import apriltag

import numpy as np

import os
import math
import sys

import traceback

#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#plt.style.use('seaborn-paper')


import cv2
have_cv2 = True
# Require cv2, do not even try PIL


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
    
import json    
from collections import OrderedDict

def detectionsToObj(detections):
    """Detection list to Python object hierarchy"""
    data = []
    for detection in detections:
        # item = {'family': detection.tag_family.decode(),
        #         'id': detection.tag_id,
        #         'hamming': detection.hamming,
        #         'goodness': detection.goodness,
        #         'decision_margin': detection.decision_margin,
        #         'H': detection.homography.tolist(),
        #         'c': detection.center.tolist(),
        #         'p': detection.corners.tolist()
        #        }
        item = OrderedDict( [
                        ('family', detection.tag_family.decode()),
                        ('id', detection.tag_id),
                        ('hamming', detection.hamming),
                        ('goodness', detection.goodness),
                        ('decision_margin', detection.decision_margin),
                        ('H', detection.homography.tolist()),
                        ('c', detection.center.tolist()),
                        ('p', detection.corners.tolist())
                    ] )
        data.append(item)
    return data
def objToDetections(data):
    """Convert Python object hierarchy to detection list"""
    detections = []
    for item in data:                
        detection = apriltag.Detection(
                item['family'].encode(),
                item['id'],
                item['hamming'],
                item['goodness'],
                item['decision_margin'],
                np.array(item['H']),
                np.array(item['c']),
                np.array(item['p']))
        detections.append(detection)
    return detections

def savejson(detections, filename):
    """Save detections to JSON file"""
    
    data = detectionsToObj(detections)
    
    #print(json.dumps(dataArray));
    #return json.dumps(dataArray)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2, sort_keys=False)
       
class Multiframejson: 
    def __init__(self, filename):
        self.filename = filename
        self.tmpfile = filename+'.tmp'
        self.beginning = True
        
    def open(self):
        with open(self.tmpfile, 'w') as outfile:
            outfile.write('{\n')
        self.beginning = True
            
    def close(self):
        with open(self.tmpfile, 'a') as outfile:
            outfile.write('}\n')
        os.rename(self.tmpfile, self.filename)
            
    def append(self, detections, frame):
        """Append detections to JSON file"""
    
        data = detectionsToObj(detections)

        with open(self.tmpfile, 'a') as outfile:
            if (not self.beginning):
                outfile.write('  ,\n')
            outfile.write('  "{f}":{{"tags":['.format(f=frame))
            
            #json.dump(data, outfile, indent=2, sort_keys=False)
            flag=False
            for item in data:
                if (flag):
                    outfile.write(',\n')
                else:
                    outfile.write('\n')
                    flag=True
            
                c=item['c']
                c[0]=float(c[0])
                c[1]=float(c[1])
                p=item['p']
                g=item['goodness']
                dm=item['decision_margin']
            
                outfile.write('      {{"id":{id},"c":[{cx:.1f},{cy:.1f}],"hamming":{hamming},"p":{corners},"g":{g},"dm":{dm}}}'.format(id=item['id'],cx=c[0],cy=c[1],hamming=item['hamming'],corners=("[[{},{}],[{},{}],[{},{}],[{},{}]]".format(p[0][0],p[0][1], p[1][0],p[1][1], p[2][0],p[2][1], p[3][0],p[3][1])),dm=dm,g=g)) 
            
            outfile.write('\n  ]}}\n'.format())
        self.beginning = False
        
def loadjson(filename):
    """Load detections from JSON file"""

    with open(filename, 'r') as infile:
        obj = json.load(infile)
    detections = objToDetections(obj)
    return detections


from timeit import default_timer as timer

def do_detect(det, orig):
    #print("do_detect: cvtColor",file=sys.stderr,flush=True)
    if len(orig.shape) == 3:
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) # Assume input is BGR
    else:
        gray = orig
    # Inversion done directly in Detector now
    #if (invert):
    #    gray = 255-gray

    #start = timer()
    #print("do_detect: det.detect",file=sys.stderr,flush=True)
    detections = det.detect(gray, return_image=False)
    #end = timer()
    #print("Time = ",end - start) 
    #print("do_detect: DONE",file=sys.stderr,flush=True)
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
    
def tagsize(D):
    def d(i,j):
        dx=D.corners[i,0]-D.corners[j,0]
        dy=D.corners[i,1]-D.corners[j,1]
        return math.sqrt(dx*dx+dy*dy)
    return max([d(0,1),d(1,2),d(2,3),d(3,0)])

# def tagradius(D):
#     x=[D.corners[i,0] for i in range(4)]
#     y=[D.corners[i,1] for i in range(4)]
#     return max([max(x)-min(x), max(y)-min(y)])
    
def draw_detections(tagimg, detections, draw_sampling=0, 
                    textthickness=2, fontscale=1.0, options=None):
    
    if (use_resource):
      print('MAXRSS draw_detections1 {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    
    for D in detections:
        #print(repr(D))
        c=D.corners
        
        def homography_project(H, x,y):
            u = H[0,0]*x+H[0,1]*y+H[0,2]
            v = H[1,0]*x+H[1,1]*y+H[1,2]
            w = H[2,0]*x+H[2,1]*y+H[2,2]
        
            return u/w,v/w
        
        if (D.tag_id<0):
            col = (0,0,255)
        else:
            if (D.hamming==0):
                col = (255,0,0)
            elif (D.hamming==1):
                col = (255,128,0)
            else:
                col = (255,255,0)
        
        if (options is not None):
          if (tagsize(D)>options.max_side_length):
              col = (0,255,255)
        
        cv2.polylines(tagimg, np.int32([c.reshape(-1, 1,2)]), False, col)
        
        st=str(D.tag_id)
        sz,baseline=cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX,fontscale,textthickness)
            
        cv2.putText(tagimg, st, tuple(np.int32([np.mean(c[:,0])-sz[0]/2,np.min(c[:,1])-5])),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale,color=col,thickness=textthickness)
        #cv2.putText(tagimg, st, tuple(np.int32([np.mean(c[:,0])-sz[0]/2,np.max(c[:,1])+15])),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale,color=col,thickness=textthickness)
        
        #cv2.putText(tagimg, "{}px".format(int(tagradius(D))), tuple(np.int32([np.mean(c[:,0])-sz[0]/2,np.max(c[:,1])+5])),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale/2,color=(0,0,255),thickness=1)
        
        if (D.tag_id>=0):
            H = D.homography;
            
            x=0; y=0;
            st="H{}".format(D.hamming)
            sz,baseline=cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX,fontscale/2,1)
            cv2.putText(tagimg, st, tuple(np.int32([np.mean(c[:,0])-sz[0]/2,np.max(c[:,1])+10])),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale/2,color=(0,0,255),thickness=1)
            
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

class FamilyPresetAction(argparse.Action):
    def __init__(self, option_strings=None, dest=None, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(FamilyPresetAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        #print('--page_size: %r %r %r' % (namespace, values, option_string))
                
        if (values.endswith('inv')):
            inverse=True
            families=values[0:-3]
        else:
            inverse=False
            families=values
                    
        setattr(namespace, 'families', families)
        setattr(namespace, 'inverse', inverse)
        
        print('Preset {}: families={}, inverse={}'.format(values, families,inverse))
        

def main():

    '''Detect apriltags.'''
    
    if (use_pympler):
      tr = tracker.SummaryTracker()

    from argparse import ArgumentParser

    # for some reason pylint complains about members being undefined :(
    # pylint: disable=E1101

    parser = ArgumentParser(
        description='Detect apriltag in videos and images. Output to tagjson/ directory')
        
    show_default = ' (default %(default)s)'
    
    apriltag.add_arguments(parser)

    parser.add_argument('filenames', metavar='IMAGE', nargs='*',
                        help='files to convert')

    parser.add_argument('-F', '--family-preset', 
                        action=FamilyPresetAction,
                        help='Family preset: detect "inv" at the end of the name to set both "family" and "inverse"')
    parser.add_argument('-V', dest='video_in', default=None,
                        help='Input video')
    parser.add_argument('-f0', dest='f0', default=0, type=int,
                        help='Frame start '+ show_default)
    parser.add_argument('-f1', dest='f1', default=0, type=int,
                        help='Frame end '+ show_default)
    parser.add_argument('-fps', dest='fps', default=20.0, type=float,
                        help='fps '+ show_default)
                        
    parser.add_argument('-tagout', dest='tagout', default=False, 
                        action='store_true',
                        help='Save image with detected tags overlay '+ show_default)
    parser.add_argument('-m', dest='multiframefile', default=False, 
                        action='store_true',
                        help='Save multiple frames into single JSON file '+ show_default)
          
    if (True):     
        parser.add_argument('-max_side_length', dest='max_side_length', 
                        default=35, type=int,
                        help='Maximum tag size in pixel '+ show_default)         
        parser.add_argument('-min_side_length', dest='min_side_length',       
                        default=20, type=int,
                        help='Tags smaller than that are discarded '+ show_default)
        parser.add_argument('-min_aspect', dest='min_aspect',
                            default=0.7, type=float,
                            help='Tags smaller than that are discarded '+ show_default)

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
    
    #init_gui()

    if not (options.video_in is None): # Input is a video
        print('Processing video {}'.format(options.video_in))
        vidcap = cv2.VideoCapture(options.video_in)
        
        if not vidcap.isOpened(): 
            print("Could not open video. Aborting.")
            raise IOError("Could not open video {}. Aborting.".format(options.video_in))
            #sys.exit(1)
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,0)     
        nframes=vidcap.get(cv2.CAP_PROP_FRAME_COUNT);
        coderfps=vidcap.get(cv2.CAP_PROP_FPS)
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,nframes-1)
        duration=vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000.0;
        
        print("Opened video.\n  nframes={}\n  nframes with cmd line fps={}\n  coder fps={}\n  coder video duration={}s={}min\n  command line fps={}\n  command line duration={}s={}min\n  time at frame {}={}s".format(nframes, nframes*coderfps/options.fps,
                 coderfps,nframes/coderfps, nframes/coderfps/60,
                 options.fps,nframes/options.fps, nframes/options.fps/60,
                 int(nframes), duration))
                 
        print('ffprobe infos:')
        import shlex, subprocess
        cmdline = "ffprobe '{}' -show_streams -loglevel -8 | grep nb_frames=".format(options.video_in)
        subprocess.call(cmdline,shell=True)
        cmdline = "ffprobe '{}' -show_streams -loglevel -8 | grep duration=".format(options.video_in)
        subprocess.call(cmdline,shell=True)
        cmdline = "ffprobe '{}' -show_streams -loglevel -8 | grep frame_rate=".format(options.video_in)
        subprocess.call(cmdline,shell=True)
        
        #options.f1=int(min(options.f1,nframes))
        
        #win = cv2.namedWindow('tags',cv2.WINDOW_NORMAL)
        
        os.makedirs('tagout',exist_ok=True)
        os.makedirs('tagjson',exist_ok=True)
        
        def printfps(t, name):
            print("Time {:10}   = {:5.3f}s   ({:4.1f} fps)".format(name, t, 1.0/t)) 
        
        fps=options.fps
        
        import gc
        
        vidcap.set(cv2.CAP_PROP_POS_MSEC,0)    
        status,orig = vidcap.read();
        # Caution: orig in BGR format by default
        if (orig is None):
            print('Warning: could not read frame {}'.format(f))
            print('Aborting...')
            raise IOError("Could not read frame {}. Aborting.".format(f))
            #sys.exit(1)
        print("Image size: {}".format(orig.shape))
        
        if (options.multiframefile):
            filenameJSON="tagjson/tags_{:05d}-{:05d}.json".format(options.f0,options.f1)
            singlejson=Multiframejson(filenameJSON)
            singlejson.open()
        
        for f in range(options.f0,options.f1+1):
        
            #vidcap.release()
            #vidcap = cv2.VideoCapture(options.video_in)
        
            filename="tagout/tagout_{:05d}.png".format(f)
            filenameJSON="tagjson/tags_{:05d}.json".format(f)
        
            print("Processing frame {}".format(f), flush=True)
        
            tstart = timer()

            vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*f)    
            status,orig = vidcap.read();
            
            # Caution: orig in BGR format by default
            if (orig is None):
                print('Warning: could not read frame {}'.format(f))
                print('Aborting...')
                break
            endread = timer()
            #printfps(endread - tstart, 'read')
            
            #print('dodetect',flush=True,file=sys.stderr)
            detections = do_detect(det, orig)
            enddetect = timer()
            #print('dodetect DONE',flush=True,file=sys.stderr)
            #printfps(enddetect-endread, 'detect')

            if (options.multiframefile):
                print("  Appending JSON to {}".format(singlejson.tmpfile))
                singlejson.append(detections, f)
            else:        
                print("  Saving JSON to {}".format(filenameJSON))
                savejson(detections, filenameJSON)

            #plot_detections(detections, gax[0], orig)
            
            if (options.tagout):
                print("  Saving tagimg to {}".format(filename))
                tagimg = orig.copy()
                draw_detections(tagimg, detections, draw_sampling=0, fontscale=0.75, options=options)            
                #tagimg = cv2.cvtColor(tagimg, cv2.COLOR_RGB2BGR);
                cv2.imwrite(filename,tagimg)
            
            #print_detections(detections, show_details=False)
            #print('Detected {} tags'.format(len(detections)))
            
            #plt.draw()
            #plt.show(block=False)
            
            #_ = input("Stopped at frame {}. Press Enter to continue".format(f))
            
            endsave = timer()
            #printfps(endsave-enddetect,'save')
            #printfps(endsave-tstart, 'TOTAL')
            if (False):
              print('  frame {:5}'.format(f))
            else:
              print('  TIME frame {:5}, {:3} tags,  time(s), {:5.3f} read, {:5.3f} detect, {:5.3f} save,  {:5.3f} total, {:4.1f} fps'.format(
                  f, len(detections), 
                  endread-tstart, enddetect-endread, endsave-enddetect, 
                  endsave-tstart, 1.0/(endsave-tstart)))
              if (use_resource):
                maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print('  MEM  frame {:5}, {:10d} bytes'.format(f, maxrss))
            
            if (use_pympler):
                tr.print_diff()
                
        if (options.multiframefile):
            print("  Closing JSON {}".format(singlejson.tmpfile))
            singlejson.close()
            
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
            
            #plt.show(block=False)

if __name__ == '__main__':

    try:
        main()
    except:
        print('apriltagdetect.py: Exception raised. See error output.')
        print('apriltagdetect.py: Exception raised:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)

