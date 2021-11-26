# import os
# import sys

# path =
# path =
# path =
# path =
# path =
# path =
# path =
# path =
# path =
# path =
# path =
# # path =
# path =''
# path = '/home/tsargsyan/saten/streaming_experimental/NeMo-nvidia/greedy_decoding.py'
# # destination = path.replace('streaming', 'streaming_experimental')
# destination = path.replace( 'streaming_experimental','streaming')
# string0 = 'mkdir '+ destination
# destination = os.path.dirname(destination)
# string = 'scp -r '+ path +' '+ destination

# if sys.argv[1] == '0':
#     string = string0

# print(string)
# os.system(string)





# import glob

# dir = '/home/tsargsyan/saten/streaming_from_scratch/streaming_experimental/NeMo-nvidia/nemo'
# all_paths = glob.glob(dir+'/**/*', recursive=True)
# for path in all_paths:
#     filename = path.split('/')[-1]
#     if '.' in filename:
#         string= 'rm ' + path
#         os.system(string)
#         print(string)

# exit()





# import os
# import sys
# import glob

# dir = '/home/tsargsyan/saten/nemo2_example/NeMo3/nemo'

# all_paths = glob.glob(dir+'/**/*', recursive=True)
# for path in all_paths:
#     path_to_copy = path.replace('nemo2_example/NeMo3', 'streaming/NeMo-nvidia')
#     if os.path.exists(path_to_copy):
#         destination = path_to_copy.replace('streaming', 'streaming_experimental')
#         string = 'scp -r '+ path +' '+ destination
#         print(string)
#         os.system(string)
