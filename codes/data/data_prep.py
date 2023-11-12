import os


import cv2
import numpy as np

# read image
img = cv2.imread('lena.jpg')
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
loc='/home/pooja/PycharmProjects/cifar/data/cifar-10-python/cifar-10-batches-py/'
for root, subdirs, files in os.walk(loc):
    print('--\nroot = ' + root)
    list_file_path = os.path.join(root, 'my-directory-list.txt')
    print('list_file_path = ' + list_file_path)
    image_number=0
    save_loc='/home/pooja/PycharmProjects/cifar/data/cv_image/'
    with open(list_file_path, 'wb') as list_file:
        #for subdir in subdirs:
            #print('\t- subdirectory ' + subdir)

        for filename in files:
            if filename.find("data_batch")>=0:dest_folder=save_loc+'/train/'
            elif filename.find("test") >=0: dest_folder=save_loc+'/test/'
            else :continue #dest_folder=save_loc+'/test/'
            file_path = os.path.join(root, filename)

            print('\t- file %s (full path: %s)' % (filename, file_path))


            f_content = unpickle(file_path)
            for i in range(len(f_content[b'labels'])):
                img = np.flip(np.swapaxes(np.swapaxes(f_content[b'data'][i].reshape((3, 32, 32)),0,2),0,1),2)
                label=f_content[b'labels'][i]

                cv2.imwrite(dest_folder+'{}_{}.jpg'.format(image_number,label), img)
                image_number+=1