import numpy as nmpy
import csv
import cv2
import h5py

def readFile(filename):
    rows = csv.reader(open(filename,"rb"))
    data = list(rows)
    for i in range(len(data)):
        data[i][1:] = [int(x) for x in data[i][1:]]
    return data

def getMaxWH(path='/home/krishna/Detection/train/'):
    max_width = 0
    max_height = 0
    for i in range(1056):
        boxes = readFile(path+'boxes_%d.txt'%i)
        for j in range(len(boxes)):
            tl_x = boxes[j][1]
            tl_y = boxes[j][2]
            br_x = boxes[j][3]
            br_y = boxes[j][4]
            if (br_x-tl_x > max_width):
                max_width = br_x-tl_x
            if (br_y-tl_y > max_height):
                max_height = br_y-tl_y

    print('max_width', max_width)
    print('max_height', max_height)

def load_data(filename, size=-1, p=0.9):

    '''

    '''
    f = h5py.File(filename, 'r')
    X = f['x_train'][:size if size!=-1 else len(f['x_train'])]
    y = f['y_train'][:size if size!=-1 else len(f['y_train'])]
    train_num = int(len(X) * p)
    return (X[:train_num], y[:train_num]), (X[train_num:], y[train_num:])

def load_test_data(filename):

    f = h5py.File(filename, 'r')
    X = f['x_test'][:]
    #ids = f['ids'][:]
    return X


# max_width = 277
# max_height = 265
def make_data(data_type):
    path = '/home/krishna/Desktop/Detection/'
    if data_type == 'train': path = path + 'train/'
    else : path = path + 'test2/'
    X = [];
    y = [];

    if data_type == 'train': num_files = 1055
    else: num_files = 126
    for num in range(num_files): #num = file number
        print (num)
        #boxes = readFile(path+"boxes_%d.txt"%num)
        imgfile = cv2.imread(path+"image_%d.png"%num)
        #for i in range(len(boxes)):
            #blank_image = nmpy.zeros((265, 277, 3), nmpy.uint8);
            #label = boxes[i][0] #label of the image
            #tl_x = boxes[i][1]
            #tl_y = boxes[i][2]
            #br_x = boxes[i][3]
            #br_y = boxes[i][4]
            #blank_image[:br_y-tl_y, :br_x-tl_x] = imgfile[tl_y:br_y, tl_x:br_x]
            #blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            #X.append(blank_image)
        X.append(imgfile)
            #y.append(label)
    
    print('converting')
    X = nmpy.array(X)
    #Y = nmpy.array(y)
    print('writing')
    if data_type == 'train':
        f = h5py.File("data/apc_train_1.hdf5", "w")
        f.create_dataset('x_train', data=X)
        #f.create_dataset('y_train', data=y)
    else:
        f = h5py.File("data/apc_test_2.hdf5", "w")
        f.create_dataset('x_test', data=X)
        #f.create_dataset('ids', data=y)
    f.close()
    print('stored')

# max_width = 277
# max_height = 265
STARNDARD_SIZE = (150, 150)
def make_data_small(data_type):
    path = '/home/krishna/Desktop/Detection/'
    if data_type == 'train': path = path + 'train/'
    else : path = path + 'test3/'
    X = [];
    y = [];

    if data_type == 'train': num_files = 1055
    else: num_files = 125
    for num in range(num_files): #num = file number
        print (num)
        boxes = readFile(path+"boxes_%d.txt"%num)
        imgfile = cv2.imread(path+"image_%d.png"%num)
        for i in range(len(boxes)):
            label = boxes[i][0] #label of the image
            tl_x = boxes[i][1]
            tl_y = boxes[i][2]
            br_x = boxes[i][3]
            br_y = boxes[i][4]
            std_img = cv2.resize(imgfile[tl_y:br_y, tl_x:br_x], STARNDARD_SIZE)
            std_img = cv2.cvtColor(imgfile, cv2.COLOR_BGR2GRAY)
            X.append(std_img)
            y.append(label)
    
    print('converting')
    X = nmpy.array(X)
    Y = nmpy.array(y)
    print('writing')
    if data_type == 'train':
        f = h5py.File("data/apc_train_1_150x150_gray.hdf5", "w")
        f.create_dataset('x_train', data=X)
        f.create_dataset('y_train', data=y)
    else:
        f = h5py.File("data/apc_test_1_150x150_gray.hdf5", "w")
        f.create_dataset('x_test', data=X)
        f.create_dataset('ids', data=y)
    f.close()
    print('stored')

    return X, y

if __name__ == '__main__':
    #print getMaxWH()
    #make_data('train')
    make_data('test')
