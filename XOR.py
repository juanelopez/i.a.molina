import math
import matplotlib.pyplot as plt

#pesos
W = [0.3 , 0.8 , -0.4 , -0.9 , 0.7 , -1 , 0.6 , -0.1 , 0.2]
b = [ -0.5 , 1 , -1 , 0.5]
ej1 = [(0,0),0]
ej2 = [(0,1),1]
ej3 = [(1,0),1]
ej4 = [(1,1),0]
ej = [ej1 , ej2 , ej3 , ej4]
w1 = []
w1.append(W[0])
w2 = []
w2.append(W[1])
w3 = []
w3.append(W[2])
w4 = []
w4.append(W[3])
w5 = []
w5.append(W[4])
w6 = []
w6.append(W[5])
w7 = []
w7.append(W[6])
w8 = []
w8.append(W[7])
w9 = []
w9.append(W[8])
b1 = []
b1.append(b[0])
b2 = []
b2.append(b[1])
b3 = []
b3.append(b[2])
b4 = []
b4.append(b[3])
LR = 0.1 #learning rate
temp = 0.00
deltaerror_1 = []
deltaerror_2 = []
deltaerror_3 = []
deltaerror_4 = []
contador = 0
#aplicamos el ejemplo a la capa de entrada
for ejemplo in ej:
    print(ejemplo[0])
    net_sigmoidea_plotting = []
    contador = contador + 1
    for _ in range(0,200):
        net1 = W[0] * ejemplo[0][0] + W[1] * ejemplo[0][1] + b[0] * 1
        net2 = W[2] * ejemplo[0][0] + W[3] * ejemplo[0][1] + b[1] * 1
        net3 = W[4] * ejemplo[0][0] + W[5] * ejemplo[0][1] + b[2] * 1
        net_sigmoidea_1 = 1/(1+math.exp(-net1))
        net_sigmoidea_2 = 1/(1+math.exp(-net2))
        net_sigmoidea_3 = 1/(1+math.exp(-net3))
        net_salida = net_sigmoidea_1 * W[6] + net_sigmoidea_2 * W[7] + net_sigmoidea_3 * W[8]
        net_salida = net_salida + b[3]        
        net_sigmoidea_salida = 1/(1+math.exp(-net_salida))
        #guardando para imprimir
        net_sigmoidea_plotting.append(net_sigmoidea_salida)
        deltaerror = net_sigmoidea_salida *(1- net_sigmoidea_salida)*(ejemplo[1]-net_sigmoidea_salida)
        #print(deltaerror)#errores
        if(contador == 1):
            deltaerror_1.append(deltaerror)
        if(contador == 2):
            deltaerror_2.append(deltaerror)
        if(contador == 3):
            deltaerror_3.append(deltaerror)
        if(contador == 4):
            deltaerror_4.append(deltaerror)
        #comienza backpropagation
        #Peso B4
        varb4 = LR * 1 * deltaerror
        b[3] = b[3] + varb4
        b4.append(b[3])
        #calcular w7 , w8 , w9
        varw7 = LR * net_sigmoidea_1 * deltaerror
        W[6] = W[6] + varw7
        w7.append(W[6])
        varw8 = LR * net_sigmoidea_2 * deltaerror
        W[7] = W[7] + varw8
        w8.append(W[7])
        varw9 = LR * net_sigmoidea_3 * deltaerror
        W[8] = W[8] + varw9
        w9.append(W[8])
        #calcular w1 , w2 , w3 , w4 , w5 , w6
        #calculo w1 y w2 y b1
        deltaminus1 = net_sigmoidea_1 *(1- net_sigmoidea_1) *(deltaerror)
        varb1 = LR * 1 * (deltaminus1)
        b[0] = b[0] + varb1
        b1.append(b[0])
        varw1 = LR * ejemplo[0][0]  * deltaminus1
        W[0] = W[0] + varw1
        w1.append(W[0])
        varw2 = LR * ejemplo[0][1] * deltaminus1
        W[1] = W[1] + varw2
        w2.append(W[1])
        #calculo w3 y w4 y b2
        deltaminus2 = net_sigmoidea_2 *(1- net_sigmoidea_2) *(deltaerror)
        varb2 = LR * 1 * (deltaminus2)
        b[1] = b[1] + varb2
        b2.append(b[1])
        varw3 = LR * ejemplo[0][0]  * deltaminus2
        W[2] = W[2] + varw3
        w3.append(W[2])
        varw4 = LR * ejemplo[0][1]* deltaminus2
        W[3] = W[3] + varw4
        w4.append(W[3])
        #calculo w5 y w6 y b3
        deltaminus3 = net_sigmoidea_3 *(1- net_sigmoidea_3) *(deltaerror)
        varb3 = LR * 1 * (deltaminus3)
        b[2] = b[2] + varb3
        b3.append(b[2])
        varw5 = LR * ejemplo[0][0]  * deltaminus3
        W[4] = W[4] + varw5
        w5.append(W[4])
        varw6 = LR * ejemplo[0][1] * deltaminus3
        W[5] = W[5] + varw6
        w6.append(W[5])
    plt.plot(net_sigmoidea_plotting)

    print("salida real:",net_sigmoidea_salida)
    print("salida esperada:" , ejemplo[1])
plt.legend([ej1[1],ej2[1],ej3[1],ej4[1]])  
plt.show()
plt.plot(deltaerror_1)
plt.plot(deltaerror_2)
plt.plot(deltaerror_3)
plt.plot(deltaerror_4)
plt.legend(["error1","error2","error3","error4"])
plt.show()
plt.plot(w1)
plt.plot(w2)
plt.plot(w3)
plt.plot(w4)
plt.plot(w5)
plt.plot(w6)
plt.plot(w7)
plt.plot(w8)
plt.plot(w9)
plt.plot(b1)
plt.plot(b2)
plt.plot(b3)
plt.plot(b4)
plt.legend(["w1","w2","w3","w4","w5","w6","w7","w8","w9","b1","b2","b3","b4"])
plt.show()
#grafico error medio
