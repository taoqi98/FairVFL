import os
import numpy as np
import imageio
import click

def load_data(path):
    images = []
    for i in range(1,202599+1):
        file = str(i)
        file = '0'*(6-len(file))+ file+'.jpg'
        im = imageio.imread(os.path.join(path,'data',file))
        images.append(im)
    images = np.array(images)
    return images

def ZipImages(images,zip_ratio=4):
    X = 218//zip_ratio
    Y = 178//zip_ratio
    ziped_images = np.zeros((len(images),X,Y,3))

    for x in range(X):
        xs = x*zip_ratio
        xe = xs + zip_ratio
        for y in range(Y):
            ys = y*zip_ratio
            ye = ys + zip_ratio

            ziped_images[:,x,y,:] = images[:,xs:xe,ys:ye,:].reshape((-1,zip_ratio**2,3)).mean(axis=1)

    ziped_images = np.array(ziped_images,dtype='int32')

    return ziped_images

def ZipImages(images,zip_ratio=4):
    X = 218//zip_ratio
    Y = 178//zip_ratio
    ziped_images = np.zeros((len(images),X,Y,3))

    for x in range(X):
        xs = x*zip_ratio
        xe = xs + zip_ratio
        for y in range(Y):
            ys = y*zip_ratio
            ye = ys + zip_ratio

            ziped_images[:,x,y,:] = images[:,xs:xe,ys:ye,:].reshape((-1,zip_ratio**2,3)).mean(axis=1)

    ziped_images = np.array(ziped_images,dtype='int32')

    return ziped_images

@click.command()
@click.option('-p', '--path')

def main(path):
    images = load_data(path)
    zipped_images = ZipImages(images)
    np.save(os.path.join(path,'zipped_images_4.npy'),zipped_images)

    cmd1 = 'cp ' + os.path.join(path,'list_attr_celeba.txt') +' ' + os.path.join(path,'attribute.txt')
    cmd2 = 'cp ' + os.path.join(path,'list_eval_partition.txt') +' ' + os.path.join(path,'eval.txt')

    os.system(cmd1)
    os.system(cmd2)

if __name__ == '__main__':
    main()
