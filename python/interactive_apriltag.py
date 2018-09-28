#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:32:17 2017

@author: megret
"""

# ipython cells

#%%

%load_ext autoreload
%autoreload 2

%matplotlib qt

cd /Users/megret/Documents/Research/BeeTracking/Soft/apriltag/swatbotics-apriltag/python

#%%
import sys

import matplotlib.pyplot as plt
import numpy as np

#%%
import cv2


folder='/Users/megret/Documents/Research/BeeTracking/Soft/labelbee-current/data/Gurabo'
video_in = folder+'/C02_170622100000.mp4'
fps = 20

vidcap = cv2.VideoCapture(video_in)

#%%

import apriltagdetect as atd

det = atd.init_detection(config='tag25h5inv')

#%%

frame = 22584

vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*frame)
status,orig = vidcap.read();

#%%
gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
rgb = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

plt.imshow(gray, cmap='gray')

#%%

tags = det.detect(gray)

#%%
import matplotlib.gridspec as gridspec
#fig = plt.figure()

plt.clf()


gs = gridspec.GridSpec(2, 2, width_ratios=[2,1])
#ax = [plt.subplot(gs_i) for gs_i in gs]
ax = [plt.subplot(gs_i) for gs_i in [gs[:,0],gs[0,1],gs[1,1]]]


ax[0].imshow(gray, cmap='gray')
atd.plot_detections(tags,ax[0], orig, labels=range(len(tags)))




id=21

pixsize = 10
S = 9*pixsize
s = pixsize
pts_src = tags[id].corners
#pts_dst = np.array([[0,0],[S,0],[S,S],[0,S]])
pts_dst = np.array([[s,s],[S-s,s],[S-s,S-s],[s,S-s]])
size = (S,S)


h, status = cv2.findHomography(pts_src, pts_dst)
im2 = cv2.warpPerspective(rgb, h, size)

ax[1].imshow(im2)

xgrid,ygrid = np.meshgrid((np.arange(9)+0.5)*s,(np.arange(9)+0.5)*s)
ax[1].plot(xgrid.ravel(),ygrid.ravel(),'+r')
ax[1].plot(pts_dst[[0,1,2,3,0],0],pts_dst[[0,1,2,3,0],1],'-g')



R = 50
S = 100
C = tags[id].corners.mean(0)
pts_src = C + np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*R
pts_dst = np.array([[0,0],[S,0],[S,S],[0,S]])
size = (S,S)

h, status = cv2.findHomography(pts_src, pts_dst)
im3 = cv2.warpPerspective(rgb, h, size)

ax[2].imshow(im3)

