import pandas as pd 
import numpy
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def data_load(file_name, col_list):
	'''
	데이터를 로드하는 함수 
	y에 해당하는 데이터는 데이터프레임 마지막에 위치해야함. 
	x, y를 반환함. 
	'''
	data = pd.read_csv(file_name, encoding = 'cp949')
	data = data[col_list]
	data=data.fillna(0)
	data_x = data.iloc[:,:-1]
	data_y = data.iloc[:,-1]
	data_y = pd.DataFrame(data_y)

	return data_x, data_y
'''
def DNN_regression(weights, biases):
	#using tensorflow 
	#depth : 4
	with tf.name_scope("dnn"):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.relu(layer_4)
        out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
        return out_layer

'''


def DNN_regression_(depth, input_size):
	'''
	using keras 
	depth는 자유조정 
	'''
	model = Sequential()
	model.add(Dense(depth, input_dim = input_size, kernel_initializer = 'normal',
		activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

def main():
	col_list = ['Play_month', 'Critic_star', 'Netizen_star_before_play', 'Num_of_participants_before_play','Expect_index_wannasee', 'Expect_index_notreally',
           'Week1_isplay','Week1_playdays','Week1_holiday','Week1_audience']
	train_x, train_y = data_load('C:/Users/JunHee/Desktop/dd/test/Train.csv', col_list)
	estimators = []
	estimators.append(('Standardize', StandardScaler()))
	estimators.append(('mip', KerasRegressor(build_fn = DNN_regression_(20,9), epochs = 100, batch_size = 5,
		verbose = 0)))
	pipeline = Pipeline(estimators) 
	seed = 7
	numpy.random.seed(seed)
	kfold = KFold(n_splits = 10, random_state = seed)
	results = cross_val_score(pipeline, train_x, train_y, cv=kfold)
	print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

if __name__ == '__main__':
	main()
