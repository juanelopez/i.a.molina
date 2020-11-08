#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:48:56 2020

@author: juane
"""
#tensorboard entendido desde https://itnext.io/how-to-use-tensorboard-5d82f8654496
#comando para inciar tensorboard tensorboard --logdir="graphs"
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_eager_execution()
#%load_ext tensorboard
tf.reset_default_graph() 
errors=[]


A=[0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1]
C=[1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1]
T=[1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]
G=[1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0]
#ENTRADEX = [[0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1],[0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0],[1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1],[0,1,1,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,0]]
A_mod = [0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1]
C_mod = [0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0]
#T_mod = [1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1]#predice mal (toma com C)
T_mod = [0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]#funciona
#T_mod = [1,1,1,1,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0]#funciona
G_mod = [0,1,1,1,0,1,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,0]#funciona (mod 1)
ENTRADEX = [A_mod,C_mod,T_mod,G_mod]
letras_x = [A,C,T,G]
letras_y = [[0,0],[0,1],[1,0],[1,1]]


x_ = tf.placeholder(tf.float32, shape=[4,25], name = 'x-input')
y_ = tf.placeholder(tf.float32, shape=[4,2], name = 'y-input')
#scalar_summary = tf.summary.scalar("X_SCALAR", x_)

# 2,3: 2 entradas para cada una de las tres neuronas en capa oculta
# 3,1: 3 entradas para una neurona en capa oculta

Pesos1 = tf.Variable(tf.random_uniform([25,9], -1, 1), name = "Pesos1")
Bias1 = tf.Variable(tf.random_uniform([9],-1,1), name = "Bias1")
Pesos2 = tf.Variable(tf.random_uniform([9,2], -1, 1), name = "Pesos2")
Bias2 = tf.Variable(tf.random_uniform([2],-1,1), name = "Bias2")


histogram_pesos1 = tf.summary.histogram('Pesos1', Pesos1)
histogram_bias1 = tf.summary.histogram('Bias1', Bias1)
histogram_pesos2 = tf.summary.histogram('Pesos2', Pesos2)
histogram_bias2 = tf.summary.histogram('Bias2', Bias2)

#scalar_pesos1 = tf.summary.scalar('Pesos1', Pesos1)
#scalar_bias1 = tf.summary.scalar('Bias1', Bias1)
#scalar_pesos2 = tf.summary.scalar('Pesos2', Pesos2)
#scalar_bias2 = tf.summary.scalar('Bias2', Bias2)

A = tf.sigmoid(tf.matmul(x_, Pesos1) + Bias1)
Salida = tf.sigmoid(tf.matmul(A, Pesos2) + Bias2)

#histogram_salida = tf.summary.histogram('Salida', Salida)
#scalar_salida = tf.summary.scalar('salida',Salida)
#Costo=tf.reduce_mean(abs(y_-Salida))
Costo=tf.reduce_mean((y_*tf.log(Salida)+((1 - y_)* tf.log(1.0-Salida)))*-1)

#histogram_costo = tf.summary.histogram('Costo', Costo)

train_step = tf.train.GradientDescentOptimizer(.9).minimize(Costo)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#t_start = time.clock()
writer = tf.summary.FileWriter('./graphs', sess.graph)
for i in range(1000):
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(train_step, feed_dict={x_: letras_x, y_: letras_y})
    errors.append(sess.run(Costo, feed_dict={x_: letras_x, y_: letras_y}))
    #summary1 ,summary2,summary3,summary4 , summary5,summary6= sess.run([histogram_pesos1,histogram_bias1,histogram_pesos2,histogram_bias2,histogram_salida,histogram_costo])
    #summary1 ,summary2,summary3,summary4 , summary5= sess.run([histogram_pesos1,histogram_bias1,histogram_pesos2,histogram_bias2,histogram_salida])
    summary1 ,summary2,summary3,summary4 = sess.run([histogram_pesos1,histogram_bias1,histogram_pesos2,histogram_bias2])
    writer.add_summary(summary1, i)
    writer.add_summary(summary2, i)
    writer.add_summary(summary3, i)
    writer.add_summary(summary4, i)
    #summary_scalar1 ,summary_scalar2,summary_scalar3,summary_scalar4 = sess.run([scalar_pesos1,scalar_bias1,scalar_pesos2,scalar_bias2])
    #summary_scalar = sess.run(scalar_summary)
    #writer.add_summary(summary_scalar, i)
    #writer.add_summary(summary_scalar1, i)
    #writer.add_summary(summary_scalar2, i)
    #writer.add_summary(summary_scalar3, i)
    #writer.add_summary(summary_scalar4, i)
    
    
    #writer.add_summary(summary5, i)
    #writer.add_summary(summary6, i)
#t_end = time.clock()
#writer.close()
print ("Serie ", i)
print ("Salida ", sess.run(Salida, feed_dict={x_: letras_x, y_: letras_y}))
print('Pesos1 ', sess.run(Pesos1))
print ('Bias1 ', sess.run(Bias1))
print('Pesos2 ', sess.run(Pesos2))
print('Bias2 ', sess.run(Bias2))
print('costo ', sess.run(Costo, feed_dict={x_: letras_x, y_: letras_y}))
#print('Tiempo transcurrido ', t_end - t_start)
plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()



#algo = [0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1]
#predictions = sess.run(pred, feed_dict={x: algo})
A = tf.sigmoid(tf.matmul(x_, Pesos1) + Bias1)
Salidax = tf.sigmoid(tf.matmul(A, Pesos2) + Bias2)
print ("Salidax ", sess.run(Salidax, feed_dict={x_: ENTRADEX, y_: letras_y}))
#histogram_salida_prueba = tf.summary.histogram('Salida prueba1', Salidax)
#sumario_prueba1= sess.run([histogram_salida_prueba])
#writer.add_summary(sumario_prueba1, 1)









