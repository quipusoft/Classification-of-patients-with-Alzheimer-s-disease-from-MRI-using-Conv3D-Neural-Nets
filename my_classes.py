
## Keras Generator for NIfti images
import numpy as np
import keras
from nilearn import image as nlimg
import nibabel as nib

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, FilePaths, labels, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=2, shuffle=True, target_shape=None, target_affine=None):
        'Initialization'
        self.FilePaths = FilePaths
        self.labels = labels
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.target_shape = target_shape
        self.target_affine = target_affine

        print('Using  Keras Generator for NIfti images V.12')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #print('\rindex:',index)
        #print('\rindexes:',indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        #print('list_IDs_temp: ',list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, Id in enumerate(list_IDs_temp):
            # Store sample
            #FA original#X[i,] = np.load('data/' + Id + '.npy')
            X[i,] = np.expand_dims(self.__NIfTI_load_image_data_by_Id(Id), axis=4)

            #print('X[i,].shape',X[i,].shape)

            # Store class
            y[i] = self.labels[Id]

        #print('X batch shape:', X.shape)
        #yp = keras.utils.to_categorical(y, num_classes=self.n_classes)

        #print('yp:',yp)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


    def __NIfTI_load_image_data_by_Id(self,Id):

        NIfTI_img = nib.load(self.FilePaths[Id])

        return  self.__NIfTI_preprocess_image(NIfTI_img)

    def __NIfTI_preprocess_image(self,img):

        # Rescale Image
        # Rotate Image
        # Resize Image
        # Flip Image
        # PCA etc.

        img = nlimg.resample_img(img, target_affine=self.target_affine, target_shape=self.target_shape, interpolation='continuous', copy=True, order='F', clip=True)

        #img is a standard NIfTI image, get_fdata() is a method of NIfTI image object. f stands for floating point array
        #return a numpy array

        return img.get_fdata()
