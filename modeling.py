import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train = pd.read_csv('C:/Users/JunHee/Desktop/dd/test/Train.csv', encoding = 'cp949')
test = pd.read_csv('C:/Users/JunHee/Desktop/dd/test/Validation.csv', encoding = 'utf8')
#모델 파라미터 튜닝을 위해 validation dataset을 test로 사용한다. 

col_list = ['Play_month', 'Critic_star', 'Netizen_star_before_play', 'Num_of_participants_before_play','Expect_index_wannasee', 'Expect_index_notreally',
           'Week1_isplay','Week1_playdays','Week1_holiday','Week1_audience']
#사용할 column은 다음과 같다. 

#변수 할당
train = train[col_list]
test = test[col_list]

X_train = train.iloc[:,:-1]
Y_train = train.iloc[:,-1]
Y_train = pd.DataFrame(Y_train)

X_test = test.iloc[:,:-1]
Y_test = test.iloc[:,-1]
Y_test = pd.DataFrame(Y_test)

total_len = X_train.shape[0]

# 모델 Parameters 세팅
learning_rate = 0.01
training_epochs = 50
batch_size = 10
display_step = 1
dropout_rate = 0.9

# Network Parameters
total_len = X_train.shape[0]
n_hidden_1 = 32 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features
n_hidden_3 = 200
n_hidden_4 = 256
n_input = X_train.shape[1]#input의 개수를 X_train 데이터에 맞춰준다.
n_classes = 1

# tf Graph input

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, 1])


# Create model : 여기서 모델은 hidden layer가 4개로 설정됨. 
def model(x, weights, biases):
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

# Store layers weight & bias
# layers의 parameter는 초기에 난수로 할당한다.(mean : 0, sd = 0.1)
with tf.name_scope("weights-biases"):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
    }

# 변수설정 및 모델 파라미터 설정 완료 

# 모델 생성
with tf.name_scope("train"):
    y_pred = model(x, weights, biases)
    cost = tf.reduce_mean(tf.square(y_pred-y))#rmse를 cost로 설정
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 모델 실행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#변수 초기화 실시

    # 모델 훈련 과정
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            print(batch_x)
            print(batch_y)
            _, c, p = sess.run([optimizer, cost, y_pred], feed_dict={x: batch_x, y: batch_y})
            print('dd')
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction

        label_value = batch_y['Week1_audience'].tolist()
        estimate = p
        err = label_value-estimate
        print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in range(10):
                print ("label value:", label_value[i], "estimated value:", estimate[i])
            print ("[*]============================")

    # Test model
    accuracy = sess.run(cost, feed_dict={x: X_test, y: Y_test})
    predicted_vals = sess.run(y_pred, feed_dict={x: X_test})
    print ("Accuracy:", accuracy)
