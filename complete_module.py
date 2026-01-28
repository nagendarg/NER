import os, sys


sys.path.append("/home/ysgnagender/NER/IE/content/IE/CRAFT/")
sys.path.insert(0, "/home/ysgnagender/NER/IE/content/IE/CRAFT/")
'''
print('Text Detection')
os.chdir('/home/ysgnagender/NER/IE/content/CRAFT/')
os.system('python3 /home/ysgnagender/NER/IE/content/CRAFT/test.py')

print('Text Recognition')
os.chdir('/home/ysgnagender/NER/IE/content/Scatter/')
os.system('python3 demo.py')
'''
print('Generating Graph Data')
os.chdir('/home/ysgnagender/NER/IE/content/Scatter/')
#os.system('python3 creat_graph_data.py')


print('Entity Extraction')
os.chdir('/home/ysgnagender/NER/IE/content/IE/')
os.system('python3 test.py --images_path ../test_image/')