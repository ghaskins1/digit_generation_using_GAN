#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:58:01 2020

@author: granthaskins
"""

import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

import os

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
#from keras.layers import Add, Subtract, Multiply, MaxPooling2D, Input, Conv2D, BatchNormalization, add, concatenate, Reshape, Lambda, UpSampling2D
from tensorflow.keras.layers import Add, Multiply, concatenate, Input, Conv2D, BatchNormalization, Reshape, UpSampling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU


class mnist_digit_generator(object):
    
    def __init__(self,epochs,batch_size,model_dir,data_dir,classGAN,training_interrupted,quad_neuron):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.classGAN = classGAN
        self.training_interrupted = training_interrupted
        self.quad_neuron = quad_neuron
        
        self.noise_shape = (100,)        
        self.optimizer = Adam(0.00005)        
            
        input_noise = Input(shape=self.noise_shape)
        
        self.generator = self.build_generator()
        
        
        if self.classGAN:
            
            (self.trainX, trainY), (self.testX, testY) = mnist.load_data()
        
            self.trainY = self.num_2_onehot(trainY)
            self.testY = self.num_2_onehot(testY)
            
            self.generator_save_terms = ['val_classGAN_generator_loss']
            
            self.classifier = self.build_classifier()
            self.classifier.compile(loss='categorical_crossentropy',
               optimizer=self.optimizer,
               metrics=['acc'])
            
            self.classifier.trainable = False
            self.classifier_save_terms = ['val_classifier_accuracy']
            
            gen_img = self.generator(input_noise)
            pred = self.classifier(gen_img)
            self.GAN = Model(inputs=input_noise,outputs=pred)
            
            self.classGAN.compile(loss='categorical_crossentropy',
                   optimizer=self.optimizer,
                   metrics=['acc'])
          
        else:
            
            (self.trainX, _), (self.testX, _) = mnist.load_data()
            

            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
               optimizer=self.optimizer,
               metrics=['acc'])
            
            self.discriminator.trainable = False
            self.discriminator_save_terms = ['val_discriminator_accuracy']
            
            self.generator_save_terms = ['val_GAN_generator_loss']
            
            gen_img = self.generator(input_noise)
            pred = self.discriminator(gen_img)
            self.GAN = Model(inputs=input_noise,outputs=pred)
            
            self.GAN.compile(loss='binary_crossentropy',
                   optimizer=self.optimizer,
                   metrics=['acc'])

        
    def num_2_onehot(self,vec):
        
        onehot = np.zeros(shape=(vec.shape[0],10))
        for i in range(onehot.shape[0]):
            
            onehot[i,vec[i]] = 1.
            
        return onehot
    
    def resblock_gen(self, num_kernels, kernel_size, x, block_num):
        
        conv1 = Conv2D(num_kernels, kernel_size, padding='same', activation=None,name='resblock_{}_conv1'.format(block_num))(x)
        BN1 = BatchNormalization(name='resblock_{}_BN1'.format(block_num))(conv1)
        act1 = Activation('relu',name='resblock_{}_act1'.format(block_num))(BN1)
        conv2 = Conv2D(num_kernels, kernel_size, padding='same', activation=None,name='resblock_{}_conv2'.format(block_num))(act1)
        BN2 = BatchNormalization(name='resblock_{}_BN2'.format(block_num))(conv2)
        add1 = Add(name='resblock_{}_add1'.format(block_num))([BN2, x])
        act2 = Activation('relu',name='resblock_{}_act2'.format(block_num))(add1)
        return act2
    
    def resblock_dis(self, num_kernels, kernel_size, x, block_num):
        conv1 = Conv2D(num_kernels, kernel_size, padding='same', activation=None,name='resblock_{}_conv1'.format(block_num))(x)
        BN1 = BatchNormalization(name='resblock_{}_BN1'.format(block_num))(conv1)
        act1 = Activation('relu',name='resblock_{}_act1'.format(block_num))(BN1)
        conv2 = Conv2D(num_kernels, kernel_size, padding='same', activation=None,name='resblock_{}_conv2'.format(block_num))(act1)
        BN2 = BatchNormalization(name='resblock_{}_BN2'.format(block_num))(conv2)
        add1 = Add(name='resblock_{}_add1'.format(block_num))([BN2, x])
        act2 = LeakyReLU(alpha=0.2)(add1)
        return act2
    
    def polynomial_block(self,input_FMs,block_num,poly_order):
        
        inputs_list = []
        FM_list = [input_FMs]
        
        for i in range(1,poly_order+1): 
            inputs_list.append(input_FMs)                     
            if i > 1:
                FM_list.append(Multiply()(inputs_list))
        output = concatenate(FM_list)
        return output

    
    def build_generator(self):
        
        noise = Input(shape=self.noise_shape,name='noise')
        dense1 = Dense(7*7*32,activation='tanh',name='dense1')(noise)
        reshape1 = Reshape((7,7,32),name='reshape1')(dense1)
        if self.quad_neuron:
            poly1 = self.polynomial_block(reshape1,1,2)
            cond_out = poly1
        else:
            cond_out = reshape1
        
        conv1 = Conv2D(32,3,strides=(1,1),padding='same',name='conv1')(cond_out)
        UpSample1 = UpSampling2D((2,2),name='UpSample1')(conv1)
        res1 = self.resblock_gen(32, 3, UpSample1, 1)
        UpSample2 = UpSampling2D((2,2),name='UpSample2')(res1)
        conv2 = Conv2D(32,3,strides=(1,1),padding='same',name='conv2')(UpSample2)
        res2 = self.resblock_gen(32, 3, conv2, 2)
        img = Conv2D(1,1,strides=(1,1),padding='same',name='img')(res2)


        return Model(inputs=noise,outputs=img)
    
    def build_discriminator(self):
    
        img = Input(shape=(28,28,1),name='img')
        conv1 = Conv2D(32,3,strides=(1,1),padding='same',name='conv1')(img)
        res1 = self.resblock_dis(32, 3, conv1, 1)
        conv2 = Conv2D(32,3,strides=(2,2),padding='same',name='conv2')(res1)
        res2 = self.resblock_dis(32, 3, conv2, 2)
        flat1 = Flatten(name='flat1')(res2)
        DO1 = Dropout(0.25,name='DO1')(flat1)
        dense1 = Dense(256,activation='relu',name='dense1')(DO1)
        DO2 = Dropout(0.25,name='DO2')(dense1)
        dense2 = Dense(128,activation='relu',name='dense2')(DO2)
        pred = Dense(1,activation='sigmoid',name='pred')(dense2)
        
        return Model(inputs=img,outputs=pred)
    
    def build_classifier(self):
    
        img = Input(shape=(28,28,1),name='img')
        conv1 = Conv2D(32,3,strides=(1,1),padding='same',name='conv1')(img)
        res1 = self.resblock_dis(32, 3, conv1, 1)
        conv2 = Conv2D(32,3,strides=(2,2),padding='same',name='conv2')(res1)
        res2 = self.resblock_dis(32, 3, conv2, 2)
        conv3 = Conv2D(32,3,strides=(2,2),padding='same',name='conv3')(res2)
        res3 = self.resblock_dis(32, 3, conv3, 3)
        flat1 = Flatten(name='flat1')(res3)
        DO1 = Dropout(0.25,name='DO1')(flat1)
        dense1 = Dense(256,activation='relu',name='dense1')(DO1)
        DO2 = Dropout(0.25,name='DO2')(dense1)
        dense2 = Dense(128,activation='relu',name='dense2')(DO2)
        pred = Dense(11,activation='softmax',name='pred')(dense2)
        
        return Model(inputs=img,outputs=pred)
    
    def data_generator(self,datatype):
        
        if datatype == 'Training':           
            idxs = np.random.randint(0,0.8*self.trainX.shape[0],self.batch_size)    
        if datatype == 'Validation':            
            idxs = np.random.randint(0.8*self.trainX.shape[0],self.trainX.shape[0],self.batch_size)           
        
        if self.classGAN:    
            
            output = [self.trainX[idxs],self.trainY[idxs]]
        else:
            
            output = self.trainX[idxs]
            
        return output
        
    def train_networks(self):
        
        steps_per_epoch = int(self.trainX.shape[0]/self.batch_size)
        
        if self.classGAN:
            
            generator_metric_names = ['classGAN_generator_loss', 'classGAN_generator_accuracy']
            val_generator_metric_names = ['val_'+item for item in generator_metric_names] 
            
            generator_empty_list = []
            for idx in range(len(generator_metric_names)):
                generator_empty_list.append([])
                
            val_generator_empty_list = []
            for idx in range(len(val_generator_metric_names)):
                val_generator_empty_list.append([])
                
            generator_loss_metric_dict = dict(zip(generator_metric_names,generator_empty_list))        
            val_generator_loss_metric_dict = dict(zip(val_generator_metric_names,val_generator_empty_list))
            
            
            classifier_metric_names = ['classifier_loss', 'classifier_accuracy']
            val_classifier_metric_names = ['val_'+item for item in classifier_metric_names]
            
            classifier_empty_list = []
            for idx in range(len(classifier_metric_names)):
                classifier_empty_list.append([])
                
            val_classifier_empty_list = []
            for idx in range(len(val_classifier_metric_names)):
                val_classifier_empty_list.append([])
                
            classifier_loss_metric_dict = dict(zip(classifier_metric_names,classifier_empty_list))        
            val_classifier_loss_metric_dict = dict(zip(val_classifier_metric_names,val_classifier_empty_list))
            
            if self.training_interrupted:
                
                self.generator = load_model(os.path.join(self.model_dir, 'classGAN_generator_current.h5'))
                self.classifier = load_model(os.path.join(self.model_dir, 'classifier_current.h5'))
                       
                for key in generator_metric_names:
    
                    train_path = os.path.join(self.data_dir, key+'.txt')
                    try:
                        generator_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                    except OSError:
                        train_path = train_path[:-4]
                        generator_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                for key in val_generator_metric_names:
                    val_path = os.path.join(self.data_dir, key+'.txt')   
                    try:
                        val_generator_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
                    except OSError:
                        val_path = val_path[:-4]
                        val_generator_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
                        
                for key in classifier_metric_names:
    
                    train_path = os.path.join(self.data_dir, key+'.txt')
                    try:
                        classifier_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                    except OSError:
                        train_path = train_path[:-4]
                        classifier_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                for key in val_classifier_metric_names:
                    val_path = os.path.join(self.data_dir, key+'.txt')   
                    try:
                        val_classifier_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
                    except OSError:
                        val_path = val_path[:-4]
                        val_classifier_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
            
            count = 0       
                                
            fake_labels = np.zeros(shape=(self.batch_size,11))
            fake_labels[:,10] = np.ones(shape=(self.batch_size,1))
            
            for epoch in range(1,self.epochs+1):
                
                generator_losses = []
                classifier_accs = []
                
                val_generator_losses = []
                val_classifier_accs = []
                
                for step in tqdm(range(steps_per_epoch),desc='Training the class GAN. Epoch {}'.format(epoch)):
                    
                    class_labels = np.zeros(shape=(2*self.batch_size,11))
                    img_arr = np.zeros(shape=(2*self.batch_size,28,28,1))
                    
                    val_class_labels = np.zeros(shape=(2*self.batch_size,11))
                    val_img_arr = np.zeros(shape=(2*self.batch_size,28,28,1))
                    
                    real_imgs,real_labels = self.data_generator('Training') 
  
                    noise = np.random.normal(0,1,(self.batch_size,self.noise_shape[0]))       
                    gen_imgs = self.generator.predict(noise)
                    
                    img_arr[:self.batch_size,:,:,0] = real_imgs
                    img_arr[self.batch_size:,:,:,0] = gen_imgs
                    class_labels[:self.batch_size,:] = real_labels
                    class_labels[:self.batch_size,:] = fake_labels
                                        
                    classifier_loss = self.classifier.train_on_batch(img_arr,class_labels)
                    classifier_accs.append(classifier_loss[1])
                                        
                    generator_loss = self.GAN.train_on_batch(noise,real_labels)
                    generator_losses.append(generator_loss[0])
                                        
                    val_real_imgs,val_real_labels = self.data_generator('Validation') 
  
                    val_noise = np.random.normal(0,1,(self.batch_size,self.noise_shape[0]))       
                    val_gen_imgs = self.generator.predict(val_noise)
                    
                    val_img_arr[:self.batch_size,:,:,0] = val_real_imgs
                    val_img_arr[self.batch_size:,:,:,0] = val_gen_imgs
                    val_class_labels[:self.batch_size,:] = val_real_labels
                    val_class_labels[:self.batch_size,:] = fake_labels
                                        
                    val_classifier_loss = self.classifier.train_on_batch(val_img_arr,val_class_labels)
                    val_classifier_accs.append(val_classifier_loss[1])
                                        
                    val_generator_loss = self.GAN.test_on_batch(val_noise,val_real_labels)
                    val_generator_losses.append(val_generator_loss[0])
                    

                    for i,key in enumerate(generator_metric_names):

                        generator_loss_metric_dict[key].append(generator_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),generator_loss_metric_dict[key])
        
                    for i,key in enumerate(val_generator_metric_names):

                        val_generator_loss_metric_dict[key].append(val_generator_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),val_generator_loss_metric_dict[key])
        

                                    
                    for term in self.generator_save_terms:
                        if 'val' in term and count > 0:
                            if val_generator_loss_metric_dict[term][-1] == min(val_generator_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.generator.save(fn_model_final)

                        elif count > 0:
                            if generator_loss_metric_dict[term][-1] == min(generator_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.generator.save(fn_model_final)
                                
                                
                    for i,key in enumerate(classifier_metric_names):

                        classifier_loss_metric_dict[key].append(classifier_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),classifier_loss_metric_dict[key])
        
                    for i,key in enumerate(val_classifier_metric_names):

                        val_classifier_loss_metric_dict[key].append(val_classifier_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),val_classifier_loss_metric_dict[key])


                    for term in self.classifier_save_terms:
                        if 'val' in term and count > 0:
                            if val_classifier_loss_metric_dict[term][-1] == min(val_classifier_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.classifier.save(fn_model_final)

                        elif count > 0:
                            if classifier_loss_metric_dict[term][-1] == min(classifier_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.classifier.save(fn_model_final)
                                
                    fn_generator = os.path.join(self.model_dir, 'classGAN_generator_current.h5')
                    self.generator.save(fn_generator)
                    
                    fn_classifier = os.path.join(self.model_dir, 'classifier_current.h5')
                    self.classifier.save(fn_classifier)
                    
                    count += 1
                    
                print(dict(zip(generator_metric_names,generator_loss)))
                print(dict(zip(classifier_metric_names,classifier_loss)))
                print(dict(zip(val_generator_metric_names,val_generator_loss)))
                print(dict(zip(val_classifier_metric_names,val_classifier_loss)))
                
                plot_step_size = 1
                    
                eps = range(steps_per_epoch)
                
                generator_losses_short = []
                val_generator_losses_short = []
                
                ep_count = 0
                for ep in eps:
                    
                    if ep % plot_step_size == 0:
                        
                        ep_count += 1
                        generator_losses_short.append(generator_losses[ep])
                        val_generator_losses_short.append(val_generator_losses[ep])
   
                plt.figure(figsize = (8,6))
                plt.title('Training and validation loss for the classGAN generator during last epoch')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                ta, = plt.plot(np.asarray(range(ep_count))*plot_step_size, generator_losses_short)
                va, = plt.plot(np.asarray(range(ep_count))*plot_step_size, val_generator_losses_short)
                plt.legend([ta, va], ['Training', 'Validation'])
                plt.show()
                
                classifier_accs_short = []
                val_classifier_accs_short = []
                
                ep_count = 0
                for ep in eps:
                    
                    if ep % plot_step_size == 0:
                        
                        ep_count += 1
                        classifier_accs_short.append(classifier_accs[ep])
                        val_classifier_accs_short.append(val_classifier_accs[ep])
   
                plt.figure(figsize = (8,6))
                plt.title('Training and validation loss for the classifier during last epoch')
                plt.xlabel('Steps')
                plt.ylabel('Accuracy')
                ta, = plt.plot(np.asarray(range(ep_count))*plot_step_size, classifier_accs_short)
                va, = plt.plot(np.asarray(range(ep_count))*plot_step_size, val_classifier_accs_short)
                plt.legend([ta, va], ['Training', 'Validation'])
                plt.show()

           
            for train_key in generator_metric_names:
                                
                val_key = 'val_' + train_key
                train_curve = generator_loss_metric_dict[train_key]
                val_curve = val_generator_loss_metric_dict[val_key]
                np.save(os.path.join(self.data_dir,train_key),np.asarray(train_curve))
                np.save(os.path.join(self.data_dir,val_key),np.asarray(val_curve))
                
                train_curve_epochs = []
                val_curve_epochs = []
                
                for i in range(len(train_curve)):
                    if i % steps_per_epoch == 0:
                        train_curve_epochs.append(train_curve[i])
                        val_curve_epochs.append(val_curve[i])
                
                eps = range(len(train_curve_epochs))
                plt.figure(figsize = (8,6))
                plt.title('Training and validation for classGAN generator ' + train_key)
                plt.xlabel('Epochs')
                plt.ylabel('Error')
                ta, = plt.plot(eps, train_curve_epochs)
                va, = plt.plot(eps, val_curve_epochs)
                plt.legend([ta, va], [train_key, val_key])
                
                filename = 'classGAN generator Training Curves '+term+'.png'
                plt.savefig(os.path.join(self.data_dir, filename), dpi=600)
                
                
            for train_key in classifier_metric_names:
                                
                val_key = 'val_' + train_key
                train_curve = classifier_loss_metric_dict[train_key]
                val_curve = val_classifier_loss_metric_dict[val_key]
                np.save(os.path.join(self.data_dir,train_key),np.asarray(train_curve))
                np.save(os.path.join(self.data_dir,val_key),np.asarray(val_curve))
                
                train_curve_epochs = []
                val_curve_epochs = []
                
                for i in range(len(train_curve)):
                    if i % steps_per_epoch == 0:
                        train_curve_epochs.append(train_curve[i])
                        val_curve_epochs.append(val_curve[i])
                
                eps = range(len(train_curve_epochs))
                plt.figure(figsize = (8,6))
                plt.title('Training and validation for classifier ' + train_key)
                plt.xlabel('Epochs')
                plt.ylabel('Error')
                ta, = plt.plot(eps, train_curve_epochs)
                va, = plt.plot(eps, val_curve_epochs)
                plt.legend([ta, va], [train_key, val_key])
                
                filename = 'classifier Training Curves '+term+'.png'
                plt.savefig(os.path.join(self.data_dir, filename), dpi=600)
 
        
        else:
            
            generator_metric_names = ['GAN_generator_loss', 'GAN_generator_accuracy']
            val_generator_metric_names = ['val_'+item for item in generator_metric_names] 
            
            generator_empty_list = []
            for idx in range(len(generator_metric_names)):
                generator_empty_list.append([])
                
            val_generator_empty_list = []
            for idx in range(len(val_generator_metric_names)):
                val_generator_empty_list.append([])
                
            generator_loss_metric_dict = dict(zip(generator_metric_names,generator_empty_list))        
            val_generator_loss_metric_dict = dict(zip(val_generator_metric_names,val_generator_empty_list))
            
            
            gen_labels = np.ones(shape=(self.batch_size,1))
            dis_labels = np.zeros(shape=(2*self.batch_size,1))
            dis_labels[:self.batch_size,0] = np.ones(shape=(self.batch_size,))
            
            discriminator_metric_names = ['discriminator_loss', 'discriminator_accuracy']
            val_discriminator_metric_names = ['val_'+item for item in discriminator_metric_names]
            
            discriminator_empty_list = []
            for idx in range(len(discriminator_metric_names)):
                discriminator_empty_list.append([])
                
            val_discriminator_empty_list = []
            for idx in range(len(val_discriminator_metric_names)):
                val_discriminator_empty_list.append([])
                
            discriminator_loss_metric_dict = dict(zip(discriminator_metric_names,discriminator_empty_list))        
            val_discriminator_loss_metric_dict = dict(zip(val_discriminator_metric_names,val_discriminator_empty_list))
            
            if self.training_interrupted:
                
                self.generator = load_model(os.path.join(self.model_dir, 'GAN_generator_current.h5'))
                self.discriminator = load_model(os.path.join(self.model_dir, 'discriminator_current.h5'))
                       
                for key in generator_metric_names:
    
                    train_path = os.path.join(self.data_dir, key+'.txt')
                    try:
                        generator_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                    except OSError:
                        train_path = train_path[:-4]
                        generator_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                for key in val_generator_metric_names:
                    val_path = os.path.join(self.data_dir, key+'.txt')   
                    try:
                        val_generator_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
                    except OSError:
                        val_path = val_path[:-4]
                        val_generator_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
                        
                for key in discriminator_metric_names:
    
                    train_path = os.path.join(self.data_dir, key+'.txt')
                    try:
                        discriminator_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                    except OSError:
                        train_path = train_path[:-4]
                        discriminator_loss_metric_dict[key] = np.loadtxt(train_path).tolist()
                for key in val_discriminator_metric_names:
                    val_path = os.path.join(self.data_dir, key+'.txt')   
                    try:
                        val_discriminator_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
                    except OSError:
                        val_path = val_path[:-4]
                        val_discriminator_loss_metric_dict[key] = np.loadtxt(val_path).tolist()
            
            count = 0            
            for epoch in range(1,self.epochs+1):
                
                generator_losses = []
                discriminator_accs = []
                
                val_generator_losses = []
                val_discriminator_accs = []
                
                for step in tqdm(range(steps_per_epoch),desc='Training the original GAN. Epoch {}'.format(epoch)):
                    
                    real_imgs = self.data_generator('Training')                    
                    img_arr = np.zeros(shape=(2*self.batch_size,28,28,1))   
                    noise = np.random.normal(0,1,(self.batch_size,self.noise_shape[0]))       
                    gen_imgs = self.generator.predict(noise)
                    
                    img_arr[:self.batch_size,:,:,0] = real_imgs
                    img_arr[self.batch_size:,:,:,:] = gen_imgs
                                        
                    discriminator_loss = self.discriminator.train_on_batch(img_arr,dis_labels)
                    discriminator_accs.append(discriminator_loss[1])
                                        
                    generator_loss = self.GAN.train_on_batch(noise,gen_labels)
                    generator_losses.append(generator_loss[0])
                    
                    
                    val_real_imgs = self.data_generator('Validation')                    
                    val_img_arr = np.zeros(shape=(2*self.batch_size,28,28,1))   
                    val_noise = np.random.normal(0,1,(self.batch_size,self.noise_shape[0]))       
                    val_gen_imgs = self.generator.predict(val_noise)
                    
                    val_img_arr[:self.batch_size,:,:,0] = val_real_imgs
                    val_img_arr[self.batch_size:,:,:,:] = val_gen_imgs
              
                    val_discriminator_loss = self.discriminator.train_on_batch(val_img_arr,dis_labels)
                    val_discriminator_accs.append(val_discriminator_loss[1])
                                        
                    val_generator_loss = self.GAN.test_on_batch(val_noise,gen_labels)
                    val_generator_losses.append(val_generator_loss[0])
                    

                    for i,key in enumerate(generator_metric_names):

                        generator_loss_metric_dict[key].append(generator_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),generator_loss_metric_dict[key])
        
                    for i,key in enumerate(val_generator_metric_names):

                        val_generator_loss_metric_dict[key].append(val_generator_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),val_generator_loss_metric_dict[key])
        

                                    
                    for term in self.generator_save_terms:
                        if 'val' in term and count > 0:
                            if val_generator_loss_metric_dict[term][-1] == min(val_generator_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.generator.save(fn_model_final)

                        elif count > 0:
                            if generator_loss_metric_dict[term][-1] == min(generator_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.generator.save(fn_model_final)
                                
                                
                    for i,key in enumerate(discriminator_metric_names):

                        discriminator_loss_metric_dict[key].append(discriminator_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),discriminator_loss_metric_dict[key])
        
                    for i,key in enumerate(val_discriminator_metric_names):

                        val_discriminator_loss_metric_dict[key].append(val_discriminator_loss[i])
                        np.savetxt(os.path.join(self.data_dir, key + '.txt'),val_discriminator_loss_metric_dict[key])


                    for term in self.discriminator_save_terms:
                        if 'val' in term and count > 0:
                            if val_discriminator_loss_metric_dict[term][-1] == min(val_discriminator_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.discriminator.save(fn_model_final)

                        elif count > 0:
                            if discriminator_loss_metric_dict[term][-1] == min(discriminator_loss_metric_dict[term]):
                                fn_model_final = os.path.join(self.model_dir, term + '.h5')                                    
                                self.discriminator.save(fn_model_final)
                                
                    fn_generator = os.path.join(self.model_dir, 'GAN_generator_current.h5')
                    self.generator.save(fn_generator)
                    
                    fn_discriminator = os.path.join(self.model_dir, 'discriminator_current.h5')
                    self.discriminator.save(fn_discriminator)
                    
                    count += 1
                    
                print(dict(zip(generator_metric_names,generator_loss)))
                print(dict(zip(discriminator_metric_names,discriminator_loss)))
                print(dict(zip(val_generator_metric_names,val_generator_loss)))
                print(dict(zip(val_discriminator_metric_names,val_discriminator_loss)))
                
                plot_step_size = 1
                    
                eps = range(steps_per_epoch)
                
                generator_losses_short = []
                val_generator_losses_short = []
                
                ep_count = 0
                for ep in eps:
                    
                    if ep % plot_step_size == 0:
                        
                        ep_count += 1
                        generator_losses_short.append(generator_losses[ep])
                        val_generator_losses_short.append(val_generator_losses[ep])
   
                plt.figure(figsize = (8,6))
                plt.title('Training and validation loss for the GAN generator during last epoch')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                ta, = plt.plot(np.asarray(range(ep_count))*plot_step_size, generator_losses_short)
                va, = plt.plot(np.asarray(range(ep_count))*plot_step_size, val_generator_losses_short)
                plt.legend([ta, va], ['Training', 'Validation'])
                plt.show()
                
                discriminator_accs_short = []
                val_discriminator_accs_short = []
                
                ep_count = 0
                for ep in eps:
                    
                    if ep % plot_step_size == 0:
                        
                        ep_count += 1
                        discriminator_accs_short.append(discriminator_accs[ep])
                        val_discriminator_accs_short.append(val_discriminator_accs[ep])
   
                plt.figure(figsize = (8,6))
                plt.title('Training and validation loss for the discriminator during last epoch')
                plt.xlabel('Steps')
                plt.ylabel('Accuracy')
                ta, = plt.plot(np.asarray(range(ep_count))*plot_step_size, discriminator_accs_short)
                va, = plt.plot(np.asarray(range(ep_count))*plot_step_size, val_discriminator_accs_short)
                plt.legend([ta, va], ['Training', 'Validation'])
                plt.show()

           
            for train_key in generator_metric_names:
                                
                val_key = 'val_' + train_key
                train_curve = generator_loss_metric_dict[train_key]
                val_curve = val_generator_loss_metric_dict[val_key]
                np.save(os.path.join(self.data_dir,train_key),np.asarray(train_curve))
                np.save(os.path.join(self.data_dir,val_key),np.asarray(val_curve))
                
                train_curve_epochs = []
                val_curve_epochs = []
                
                for i in range(len(train_curve)):
                    if i % steps_per_epoch == 0:
                        train_curve_epochs.append(train_curve[i])
                        val_curve_epochs.append(val_curve[i])
                
                eps = range(len(train_curve_epochs))
                plt.figure(figsize = (8,6))
                plt.title('Training and validation for GAN generator ' + train_key)
                plt.xlabel('Epochs')
                plt.ylabel('Error')
                ta, = plt.plot(eps, train_curve_epochs)
                va, = plt.plot(eps, val_curve_epochs)
                plt.legend([ta, va], [train_key, val_key])
                
                filename = 'GAN generator Training Curves '+train_key+'.png'
                plt.savefig(os.path.join(self.data_dir, filename), dpi=600)
                
                
            for train_key in discriminator_metric_names:
                                
                val_key = 'val_' + train_key
                train_curve = discriminator_loss_metric_dict[train_key]
                val_curve = val_discriminator_loss_metric_dict[val_key]
                np.save(os.path.join(self.data_dir,train_key),np.asarray(train_curve))
                np.save(os.path.join(self.data_dir,val_key),np.asarray(val_curve))
                
                train_curve_epochs = []
                val_curve_epochs = []
                
                for i in range(len(train_curve)):
                    if i % steps_per_epoch == 0:
                        train_curve_epochs.append(train_curve[i])
                        val_curve_epochs.append(val_curve[i])
                
                eps = range(len(train_curve_epochs))
                plt.figure(figsize = (8,6))
                plt.title('Training and validation for discriminator ' + train_key)
                plt.xlabel('Epochs')
                plt.ylabel('Error')
                ta, = plt.plot(eps, train_curve_epochs)
                va, = plt.plot(eps, val_curve_epochs)
                plt.legend([ta, va], [train_key, val_key])
                
                filename = 'discriminator Training Curves '+train_key+'.png'
                plt.savefig(os.path.join(self.data_dir, filename), dpi=600)
                
                
    def generate_imgs_w_GAN(self,save_term,num_imgs):
        
        if self.classGAN:
            self.generator = load_model(os.path.join(self.model_dir, 'classGAN_generator_'+save_term+'.h5'))
        else:
            self.generator = load_model(os.path.join(self.model_dir, 'GAN_generator_'+save_term+'.h5'))
                
        noise = np.random.normal(0,1,(num_imgs,self.noise_shape[0]))       
        gen_imgs = self.generator.predict(noise)
        
        if len(gen_imgs.shape) == 4:
            
            gen_imgs = gen_imgs[:,:,:,0]
        
        for i in range(1,gen_imgs.shape[0]+1):
            
            plt.imshow(gen_imgs[i])
            plt.axis('off')
            plt.title('Generated image {}'.format(i))
            plt.show()

           
    
digit_generator = mnist_digit_generator(epochs=5,batch_size=512,model_dir='/Users/granthaskins/Downloads/model_dir',data_dir='/Users/granthaskins/Downloads/data_dir',classGAN=False,training_interrupted=False,quad_neuron=False)   
digit_generator.train_networks()    
        

