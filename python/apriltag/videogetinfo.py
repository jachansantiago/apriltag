#!/usr/bin/env python

""" Simple util to populate video info file

"""

import os, sys
import json
import shlex, subprocess

from ffprobe3 import FFProbe
import ffprobe3

import copy
import pprint

pp = pprint.PrettyPrinter(indent=4).pprint


from distutils.util import strtobool

def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')



def main():

    '''Populate video info file.'''
    
    from argparse import ArgumentParser

    # for some reason pylint complains about members being undefined :(
    # pylint: disable=E1101

    parser = ArgumentParser(
        description='Run ffprobe on file to find video FPS. Load and restore to videoname.mp4.info')
        
    show_default = ' (default %(default)s)'

    parser.add_argument('-V', dest='video_in', default=None,
                        help='Input video')

    parser.add_argument('-F', '--family-preset', dest='familyPreset',
                        help='Family preset: detect "inv" at the end of the name to set both "family" and "inverse"')
    parser.add_argument('-fps', dest='fps', default=20.0, type=float,
                        help='video codec FPS, may be different to actual fps for broken encoders '+ show_default)
                        
    options = parser.parse_args()
    
    if (options.video_in == None):
        print('No video provided')
        return -1
    if (not os.path.isfile(options.video_in)):
        print('Videofile {} not found'.format(options.video_in))
        return -1
    
    jsonfile = options.video_in+'.info'
    jsonfileout=jsonfile #+'_out.txt'
    
    print("jsonfile="+jsonfile)
    print("jsonfileout="+jsonfileout)
    
    if (os.path.isfile(jsonfile)):
        with open(jsonfile) as infile:
            try:
              data_orig = json.load(infile)
              data = copy.deepcopy(data_orig)
            except ValueError as err:
              print('Error parsing JSON file {}'.format(jsonfile))
              print(err)
              return 1
        print("Loaded old JSON file: {}".format(jsonfile))
        print(data)
    else:
        data = {'place': 'Gurabo'}
        print("No preexisting JSON file: {}".format(jsonfile))
        print(data)
        
    metadata=FFProbe(options.video_in)
    
    #FFStream.frame_rate = get_frame_rate

    def get_field(self, name):
        """
        Returns named parameter using ffprobe.
        Returns none is it is not a video stream.
        """
        f = None
        if self.is_video():
            if self.__dict__[name]:
                f = self.__dict__[name]
        return f

    for stream in metadata.streams:
        if stream.is_video():
        
            #print(stream.__dict__.keys())
        
            data["nframes"]=stream.frames()
            data["duration"]=stream.duration_seconds()
            
            r_frame_rate=get_field(stream,'r_frame_rate')
            ratio = r_frame_rate.split('/')
            if (len(ratio)==1):
                fps = float(ratio[0])
            elif (len(ratio)==2):
                fps = float(ratio[0])/float(ratio[1])
            else:
                print('Error, could not parse r_frame_rate={}'.format(r_frame_rate))

            data["r_frame_rate"]=r_frame_rate            
            data["videofps"]=fps
        
    #data["ffprobe"] = stream.__dict__
    
    print("### Original JSON:")
    pp(data_orig)
    print("### Updated JSON:")
    pp(data)
    
    if (not user_yes_no_query('Update file {} ?'.format(jsonfileout))):
        return 0
    
    with open(jsonfileout, 'w') as outfile:
        json.dump(data, outfile, indent=2, sort_keys=False)
        outfile.write('\n')
        
    return 0

if __name__ == '__main__':

    code = main()
    sys.exit(code)