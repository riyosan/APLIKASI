from keras.engine import Layer
from keras.layers import Activation, Flatten
from keras.layers import Input,InputLayer, Activation, Convolution2D, MaxPooling2D, Flatten, Dense, Conv2D, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import optimizers, layers, callbacks
import keras.backend as K
import keras
from keras import layers, models, optimizers


class Capsule(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=10, kernel_size=(9, 1), share_weights=False,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# def build_model(lr):
#     hidden_num_units = 5
#     output_num_units = 3
#     model = Sequential([
#     InputLayer(input_shape=(224,224,1)),
#     #layer1
#     Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding="valid", activation="relu"),
#     MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
#     #layer2
#     Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid", activation="relu"),
#     MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
#     #layer 3-5
#     Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
#     Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
#     Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
#     MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
    
#     Capsule(num_capsule=32, dim_capsule=16, routings=10, kernel_size=(9, 1), share_weights=False),
    
#     Flatten(),

#     Dense(units=hidden_num_units, activation='tanh'),

#     Dense(units=hidden_num_units, activation='tanh'),

#     Dense(units=hidden_num_units, activation='tanh'),

#     Dense(units=hidden_num_units, activation='tanh'),

#     Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax')
#     ])
#     if lr!=0:
#         model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.Adadelta(lr=lr), metrics=["accuracy"])
#     else:
#         model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.Adadelta(rho=0.95), metrics=["accuracy"])
    
#     return model

def build_model(lr):
    input_reshape = (28,28,1)
    pool_size = (2, 2)
    hidden_num_units = 5
    output_num_units = 3
    model = Sequential([
    InputLayer(input_shape=input_reshape),

    Conv2D(25, 5, 5, activation='relu'),
    MaxPooling2D(pool_size=pool_size),

    Conv2D(16, 5, 5, activation='relu'),
    MaxPooling2D(pool_size=pool_size),

    Conv2D(25, 4, 4, activation='relu'),

    Capsule(num_capsule=16, dim_capsule=8, routings=3,share_weights=False),

    Flatten(),

    Dense(units=hidden_num_units, activation='tanh'),

    Dense(units=hidden_num_units, activation='tanh'),

    Dense(units=hidden_num_units, activation='tanh'),

    Dense(units=hidden_num_units, activation='tanh'),

    Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax')
    ])
    if lr!=0:
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.Adadelta(lr=lr), metrics=["accuracy"])
    else:
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.Adadelta(rho=0.95), metrics=["accuracy"])
    
    return model