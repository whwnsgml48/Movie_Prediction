import pandas as pd
import numpy as np
import tensorflow as tf


def data_load(file_name, col_list):
    '''
	데이터를 로드하는 함수 
	y에 해당하는 데이터는 데이터프레임 마지막에 위치해야함. 
	x, y를 반환함. 
	'''
    data = pd.read_csv(file_name, encoding='cp949')
    data = data[col_list]
    data = data.fillna(0)
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    data_y = pd.DataFrame(data_y)

    return data_x, data_y


def DNN_regression(x, weights, biases):
    # using tensorflow
    # depth : 4
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


def main():
    col_list = ['Play_month', 'Critic_star', 'Netizen_star_before_play', 'Num_of_participants_before_play',
                'Expect_index_wannasee', 'Expect_index_notreally',
                'Week1_isplay', 'Week1_playdays', 'Week1_holiday', 'Week1_audience']
    X_train, Y_train = data_load('C:/Users/JunHee/Desktop/dd/test/Train.csv', col_list)
    X_test, Y_test = data_load('C:/Users/JunHee/Desktop/dd/test/Train.csv', col_list)

    # 모델 Parameters 세팅
    learning_rate = 0.01
    training_epochs = 50
    batch_size = 10
    display_step = 1
    dropout_rate = 0.9

    # Network Parameters
    total_len = X_train.shape[0]
    n_hidden_1 = 32  # 1st layer number of features
    n_hidden_2 = 200  # 2nd layer number of features
    n_hidden_3 = 200
    n_hidden_4 = 256
    n_input = X_train.shape[1]  # input의 개수를 X_train 데이터에 맞춰준다.
    n_classes = 1

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])

    with tf.name_scope('weight-bias'):
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

    # 모델 생성
    with tf.name_scope("train"):
        y_pred = DNN_regression(x, weights, biases)
        cost = tf.reduce_mean(tf.square(y_pred - y))  # rmse를 cost로 설정
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 모델 실행
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 변수 초기화 실시

        # 모델 훈련 과정
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x = X_train[i * batch_size : (i + 1) * batch_size]
                batch_y = Y_train[i * batch_size : (i + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, y_pred], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # sample prediction

            label_value = batch_y['Week1_audience'].tolist()
            estimate = p
            err = label_value - estimate
            print("num batch:", total_batch)

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print("[*]----------------------------")
                for i in range(10):
                    print("label value:", label_value[i], "estimated value:", estimate[i])
                print("[*]============================")

        # Test model
        accuracy = sess.run(cost, feed_dict={x: X_test, y: Y_test})
        predicted_vals = sess.run(y_pred, feed_dict={x: X_test})
        print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()
