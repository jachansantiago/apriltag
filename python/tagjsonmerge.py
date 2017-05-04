#import apriltagdetect as atd
import json


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
    #parser.add_argument('-fps', dest='fps', default=22.0, type=float,
    #                    help='fps '+ show_default)
                        
    parser.add_argument('-tags', dest='tags', 
                        default="tagjson/tags_{:05d}.json",
                        help='tags file pattern '+ show_default)
    parser.add_argument('-o', dest='output', 
                        default="tags_{f0:05d}_{f1:05d}.json",
                        help='out file '+ show_default)
                            
    options = parser.parse_args()
                            
    if (options.f1<options.f0):
        options.f1=options.f0
    
    print(options)
    
    outname=options.output.format(f0=options.f0,f1=options.f1)
    
    print('Processing input JSON with pattern {}'.format(options.tags))
    print('Output {}'.format(outname))
        
    with open(outname, 'w') as outfile:
        outfile.write("{\n")
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
                        # print(item)
                        # detection = apriltag.Detection(
                        #     item['family'].encode(),
                        #     item['id'],
                        #     item['hamming'],
                        #     item['goodness'],
                        #     item['decision_margin'],
                        #     np.array(item['H']),
                        #     np.array(item['c']),
                        #     np.array(item['p']))
                    
                        c=item['c']
                        c[0]=float(c[0])
                        c[1]=float(c[1])
                        p=item['p']
                    
                        outfile.write('      {{"id":{id},"c":[{cx:.1f},{cy:.1f}],"hamming":{hamming},"p":{corners}}}'.format(id=item['id'],cx=c[0],cy=c[1],hamming=item['hamming'],corners=("[[{},{}],[{},{}],[{},{}],[{},{}]]".format(p[0][0],p[0][1], p[1][0],p[1][1], p[2][0],p[2][1], p[3][0],p[3][1])))) 
                    print('Added {} tags for frame {}: {}'.format(len(obj),f,[item['id'] for item in obj]))
            except IOError as e:
                print('Could not open {}, ignoring.'.format(filenameJSON))
                    
            outfile.write("\n    ]\n  }\n")               
            if (f<options.f1): outfile.write("  ,\n");
        outfile.write("}\n")


if __name__ == '__main__':

    main()
    