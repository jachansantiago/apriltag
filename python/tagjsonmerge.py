#!/usr/bin/env python

#import apriltagdetect as atd
import json
from collections import OrderedDict
import os

def main():

    '''Consolidate apriltags into simple JSON.'''

    from argparse import ArgumentParser

    # for some reason pylint complains about members being undefined :(
    # pylint: disable=E1101

    parser = ArgumentParser(
        description='Consolidate apriltags into simple JSON.')
        
    show_default = ' (default %(default)s)'

    parser.add_argument('-f0', dest='f0', default=0, type=int,
                        help='Frame start '+ show_default)
    parser.add_argument('-f1', dest='f1', default=0, type=int,
                        help='Frame end '+ show_default)
    parser.add_argument('-ms', dest='multiframestep', default=0, type=int,
                        help='If >0, input is multiple frame files with step '+ show_default)
    #parser.add_argument('-fps', dest='fps', default=22.0, type=float,
    #                    help='fps '+ show_default)
                        
    parser.add_argument('-tags', dest='tags', 
                        default="tagjson/tags_{:05d}.json",
                        help='tags file pattern '+ show_default)
    parser.add_argument('-mtags', dest='mtags', 
                        default="tagjson/tags_{:05d}-{:05d}.json",
                        help='multiframe tags file pattern (used with -ms) '+ show_default)
    parser.add_argument('-o', dest='output', 
                        default="tags_{f0:05d}_{f1:05d}.json",
                        help='out file '+ show_default)
                            
    options = parser.parse_args()
                            
    if (options.f1<options.f0):
        options.f1=options.f0
    
    print(options)
    
    outname=options.output.format(f0=options.f0,f1=options.f1)
    
    print('Input JSON pattern: {}'.format(options.tags))
    print('Output:             {}'.format(outname))
        
    with open(outname, 'w') as outfile:
        outfile.write("{\n")
        
        if (options.multiframestep==0): # Single frame input
            for f in range(options.f0,options.f1+1):
                outfile.write('  "{}":{{"tags":['.format(f))
                filenameJSON=options.tags.format(f)
                try:
                    with open(filenameJSON, 'r') as infile:
                        obj = json.load(infile)
                        flag=False
                        for item in obj:
                            if (flag):
                                outfile.write(',\n')
                            else:
                                outfile.write('\n')
                                flag=True
                    
                            c=item['c']
                            c[0]=float(c[0])
                            c[1]=float(c[1])
                            p=item['p']
                            dm=item['decision_margin']
                    
                            outfile.write('      {{"id":{id},"c":[{cx:.2f},{cy:.2f}],"hamming":{hamming},"p":{corners},"dm":{dm}}}'.format(id=item['id'],cx=c[0],cy=c[1],hamming=item['hamming'],corners=("[[{},{}],[{},{}],[{},{}],[{},{}]]".format(p[0][0],p[0][1], p[1][0],p[1][1], p[2][0],p[2][1], p[3][0],p[3][1])),dm=dm)) 
                        print('Added {} tags for frame {}: {}'.format(len(obj),f,[item['id'] for item in obj]))
                except IOError as e:
                    print('Could not open {}, ignoring.'.format(filenameJSON))
                    
                outfile.write("\n    ]\n  }\n")               
                if (f<options.f1): outfile.write("  ,\n");
        elif (options.multiframestep>=1): # Multiple frame input
            
            mergesources = []
            for fb in range(options.f0,options.f1+1,options.multiframestep):
                filenameJSON=options.mtags.format(fb,
                                                  fb+options.multiframestep-1)
                mergesources.append(filenameJSON)
                
            print('Sources for the merge:')
            for i,filenameJSON in enumerate(mergesources):
                exist = os.path.isfile(filenameJSON) 
                print("{:03} {} {}".format(i,filenameJSON, "exist="+str(int(exist)) ))
            
            # Copy meta info from first file
            fb = options.f0
            filenameJSON=mergesources[0]
            with open(filenameJSON, 'r') as infile:
                fileobj = json.load(infile, object_pairs_hook=OrderedDict)
                
                info = fileobj['info']
                
                info['mergesources'] = mergesources
                
                outfile.write('"info":')
                json.dump(info, outfile, indent=2, sort_keys=False)
                outfile.write('\n')
                
                outfile.write(',\n"data":{')
            
            for filenameJSON in mergesources:
                
                try:
                    with open(filenameJSON, 'r') as infile:
                        fileobj = json.load(infile, object_pairs_hook=OrderedDict)
                        
                        data = fileobj['data']
                        
                        isFirstItem=True
                        
                        for frame in data.keys():
                            framedata = data[frame]
                            
                            frametags = framedata['tags']
                            N = len(frametags)
                            
                            if (not isFirstItem):
                                outfile.write('  ,\n')
                            else:
                                outfile.write('\n')
                                isFirstItem=False

                            outfile.write('  "{}":'.format(frame))
                            
                            outfile.write('{"tags": [')
                            tags=[]
                            for tag in frametags:
                                tags.append(json.dumps(tag, indent=None, sort_keys=False))
                            if (len(frametags)>0):
                                outfile.write('\n    ' + ',\n    '.join(tags))
                            outfile.write('\n  ]}') # end of tags, end of frame

                            print('Added {} tags for frame {}: {}'.format(N,frame,[str(item['id']) for item in frametags]))
                except IOError as e:
                    print('Could not open {}, ignoring.'.format(filenameJSON))
            
            outfile.write('\n}\n\n')  # end of data
            
        else:
            print("ERROR: options.multiframestep should be positive")
        outfile.write("}\n")  # end of root


if __name__ == '__main__':

    main()
    
