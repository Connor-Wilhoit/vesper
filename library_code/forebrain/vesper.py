N_img_height   = 66
N_img_width    = 220
N_img_channels = 3

def vesper():
    input_shape = (N_img_height, N_img_width, N_img_channels)
    model = Sequential()
    # normalization - perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape = input_shape))
        
    # First Convolutional Layer
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid',
                                        kernel_initializer='he_normal', name='conv1'))
    # Exponential Linear Unit #1
    model.add(ELU())
    
    # Second Convolutional Layer
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid',
                                        kernel_initializer='he_normal', name='conv2'))
    # Exponential Linear Unit #2
    model.add(ELU())
    
    # Third Convolutional Layer
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid',
                                        kernel_initializer='he_normal', name='conv3'))
    # Exponentail Linear Layer #3, followed by our first Dropout Layer
    model.add(ELU())
    model.add(Dropout(0.5))
    
    # Fourth Convolutional Layer
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='valid',
                                        kernel_initializer='he_normal', name='conv4'))
    # Exponential Linear Unit #4
    model.add(ELU())
    
    # Fifth Convolutional Layer
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='valid',
                                        kernel_initializer='he_normal', name='conv5'))

    # Now we'll Flatten out vesper, and add some Dense-ass layers ;)
    model.add(Flatten(name='flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
    model.add(ELU())
    
    # we're NOT going to put an activation OR a non-linearity at the end of our
    # convolutional neural network (cnn), and the reason for this is that we want
    # EXACT ouput, NOT a class identifier.
    model.add(Dense(1, name='output', kernel_initializer='he_normal'))
    
    # Finally, we'll get vesper's optimizer prepared, and fed to Her
    # note: I'm going to tryout `AdamW` as opposed to the traditional `Adam`, and see what happens!
    #adam = AdamW(lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, amsgrad=False)
    
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    
    return model
