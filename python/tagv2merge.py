#import apriltagdetect as atd
import json
from collections import OrderedDict
import os

def main():

    '''Consolidate several apriltags JSON into one JSON.'''

    from argparse import ArgumentParser

    # for some reason pylint complains about members being undefined :(
    # pylint: disable=E1101

    parser = ArgumentParser(
        description='Consolidate apriltags into simple JSON.')
        
    show_default = ' (default %(default)s)'

    parser.add_argument('files', metavar='N', type=str, nargs='+',
                    help='list of input files')
                        
    parser.add_argument('-o', dest='output', 
                        default="mergedtags_{f0:05d}_{f1:05d}.json",
                        help='out file '+ show_default)
    parser.add_argument('-v', dest='verbose', 
                        default=False, action='store_true',
                        help='Verbose mode '+ show_default)
                            
    options = parser.parse_args()
                            
    print(options)
    
    mergesources = options.files
    
    info = None
    allframes = []
    alldata = {}
    for filenameJSON in options.files:
        print("Loading input file {}".format(filenameJSON))
        with open(filenameJSON, 'r') as infile:
            obj = json.load(infile)
            
            if (info is None):
                info = obj['info']
            data = obj['data']
            
            frames = data.keys()
            alldata.update(data)
    allframes = sorted([int(f) for f in alldata.keys()])

    #print(allframes)
    
    f0 = int(allframes[0])
    f1 = int(allframes[-1])
    print("f0={}, f1={}".format(f0,f1))
    
    outname = options.output.format(f0=f0,f1=f1)
    print("outname={}".format(outname))
    
    missing = []
    if allframes[0]>0:
        missing.append('{}-{}'.format(0, allframes[0]-1))
    for i in range(1,len(allframes)):
       if allframes[i]>allframes[i-1]+1:
           missing.append('{}-{}'.format(allframes[i-1]+1, allframes[i]-1))
    print('Missing: {}'.format(','.join(missing)))

    with open(outname, 'w') as outfile:
        outfile.write("{\n")

        if ('log' not in info):
            info['log']=[]
        info['log'].append({"description":"Merged tag files",
                            "config":vars(options),
                            "mergesources":mergesources})
         
        outfile.write('"info":')
        json.dump(info, outfile, indent=2, sort_keys=False)
        outfile.write('\n')
            
        outfile.write(',\n"data":{')
                
        isFirstItem=True
        counts = {}
        fcounts = {}
        count = 0
                          
        for frame in alldata.keys():
            framedata = alldata[frame]
            
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
            if (N>0):
                outfile.write('\n    ' + ',\n    '.join(tags))
            outfile.write('\n  ]}') # end of tags, end of frame

            #id = item['id']
            #for item in frametags:
            #    counts[id]=counts.get(id,0)+1
            #fcounts[frame]=N
            count+=N

        print('  Added {} tags:'.format(count))
        
        if (options.verbose):
            #print('  By frame: '+str(fcounts))
            #print('  By id: '+str(counts))
            pass
          
        outfile.write('\n}\n\n')  # end of data
        
        outfile.write("}\n")  # end of root


if __name__ == '__main__':

    main()
    
