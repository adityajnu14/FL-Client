import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf  
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras import backend as K


# #Fetech locally saved(Device level) model
# def init():
    
#     K.clear_session()
#     model = load_model('Models/model.h5')
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model

# #Create and initilize model for first time. 
# def freshModel():
#     K.clear_session()
#     model = Sequential([
#         Dense(units=2, input_shape=(4, ), activation='relu'),
#         #Dense(units=8,  activation='relu'),
#         Dense(units=1, activation='sigmoid')
#     ])

#     model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
#     model.save('Models/InitModel.h5')


#Fetech locally saved(Device level) model
def init():
    
    K.clear_session()
    model = load_model('Models/model.h5')
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

#Create and initilize model for first time. 
def freshModel():
    K.clear_session()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(512,)))
    #model.add(tf.keras.layers.Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='sigmoid'))


    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.save('Models/InitModel.h5')


