from distutils.log import debug
from fileinput import filename
from flask import *
import boto3

import numpy 
from inference import Video,Decoder,labels_to_text,Spell,tokenizer,token

import os
import matplotlib
import tensorflow as tf
import keras
tf.compat.v1.disable_eager_execution() 
tf.compat.v1.experimental.output_all_intermediates(True)
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.core import Lambda
from scipy import ndimage
import numpy as np
import sys
import os
from keras import backend as K
import numpy as np
import math
# import the fuzzywuzzy module
from fuzzywuzzy import fuzz

app = Flask(__name__)



# AWS S3 configuration
S3_BUCKET = 'aws-project-bucket-2'
S3_ACCESS_KEY = 'AKIAQ5YBX4K7LHSGGDCZ'
S3_SECRET_KEY = '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'
S3_LOCATION = 'http://{}.s3.amazonaws.com/'.format(S3_BUCKET)

s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def inference(file):
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	#es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1,patience=6)
	#mc = ModelCheckpoint('drive/MyDrive/lipnetWthSil/best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss

	print(os.path.abspath(os.getcwd()))
	current_file_path=os.path.join(os.path.abspath(os.getcwd()),'cloud-project')
	weights_directory=os.path.join(current_file_path,'weights-lipreading')
	weights_path=os.path.join(weights_directory,'best_model.h5')
	print("Weights_paths: ",weights_path)
	model= keras.models.load_model(weights_path,compile=False)
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=['accuracy'])
	prediction_model = keras.models.Model(model.get_layer(name="the_input").input, model.get_layer(name="softmax").output)

	CURRENT_PATH = os.path.dirname('D:\\flask_app_upload\\')
	PREDICT_GREEDY      = False
	PREDICT_BEAM_WIDTH  = 200
	PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'dictionaries','urdu_sentences.txt')
	absolute_max_string_len=29
	output_size=43

	video=Video().from_frames(file)
	spell = Spell(path=PREDICT_DICTIONARY)
	decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                          postprocessors=[labels_to_text, spell.sentence])

	X_data       = np.array([video.data]).astype(np.float32) /255
	input_length = np.array([len(video.data)])
  
	y_pred         = prediction_model.predict(X_data)
	result         = decoder.decode(y_pred, input_length)
	new_result=+'میں'+" "+result[0]
	#new_result=token(new_result)
 	#new_result = tokenizer(result[0])

	print("Result: ",new_result)

	return new_result.encode("utf-8") 

    
@app.route('/')
def main():
	return render_template("index.html")

@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		print(f.filename)
		result=inference(f.filename)
		return result

if __name__ == '__main__':
	app.run(debug=True)
