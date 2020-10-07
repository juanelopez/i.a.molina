import math
#pesos
W = [0.3 , 0.8 , -0.4 , -0.9 , 0.7 , -1 , 0.6 , -0.1 , 0.2]
b = [ -0.5 , 1 , -1 , 0.5]
ej1 = [(0,0),0]
ej2 = [(0,1),1]
ej3 = [(1,0),1]
ej4 = [(1,1),0]
ej = [ej1 , ej2 , ej3 , ej4]
LR = 0.1 #learning rate
#aplicamos el ejemplo a la capa de entrada
for ejemplo in ej:
    print(ejemplo[0])
    for _ in range(0,10000):
        net1 = W[0] * ejemplo[0][0] + W[1] * ejemplo[0][1] + b[0] * 1
        net2 = W[2] * ejemplo[0][0] + W[3] * ejemplo[0][1] + b[1] * 1
        net3 = W[4] * ejemplo[0][0] + W[5] * ejemplo[0][1] + b[2] * 1
        net_sigmoidea_1 = 1/(1+math.exp(-net1))
        net_sigmoidea_2 = 1/(1+math.exp(-net2))
        net_sigmoidea_3 = 1/(1+math.exp(-net3))
        net_salida = net_sigmoidea_1 * W[6] + net_sigmoidea_2 * W[7] + net_sigmoidea_3 * W[8]
        net_salida = net_salida + b[3]
        net_sigmoidea_salida = 1/(1+math.exp(-net_salida))
        deltaerror = net_sigmoidea_salida *(1- net_sigmoidea_salida)*(ejemplo[1]-net_sigmoidea_salida)
        #comienza backpropagation
        #Peso B4
        varb4 = LR * 1 * deltaerror
        b[3] = b[3] + varb4
        #calcular w7 , w8 , w9
        varw7 = LR * net_sigmoidea_1 * deltaerror
        W[6] = W[6] + varw7
        varw8 = LR * net_sigmoidea_2 * deltaerror
        W[7] = W[7] + varw8
        varw9 = LR * net_sigmoidea_3 * deltaerror
        W[8] = W[8] + varw9
        #calcular w1 , w2 , w3 , w4 , w5 , w6
        #calculo w1 y w2
        deltaminus1 = net_sigmoidea_1 *(1- net_sigmoidea_1) *(deltaerror)
        varb1 = LR * 1 * (deltaminus1)
        b[0] = b[0] + varb1
        varw1 = LR * ejemplo[0][0]  * deltaminus1
        W[0] = W[0] + varw1
        varw2 = LR * ejemplo[0][1] * deltaminus1
        W[1] = W[1] + varw2
        #calculo w3 y w4
        deltaminus2 = net_sigmoidea_2 *(1- net_sigmoidea_2) *(deltaerror)
        varb2 = LR * 1 * (deltaminus2)
        b[1] = b[1] + varb2
        varw3 = LR * ejemplo[0][0]  * deltaminus2
        W[2] = W[2] + varw3
        varw4 = LR * ejemplo[0][1]* deltaminus2
        W[3] = W[3] + varw4
        #calculo w5 y w6
        deltaminus3 = net_sigmoidea_3 *(1- net_sigmoidea_3) *(deltaerror)
        varb3 = LR * 1 * (deltaminus3)
        b[2] = b[2] + varb3
        varw5 = LR * ejemplo[0][0]  * deltaminus3
        W[4] = W[4] + varw3
        varw6 = LR * ejemplo[0][1] * deltaminus3
        W[5] = W[5] + varw4
    print("salida real:",net_sigmoidea_salida)
    print("salida esperada:" , ejemplo[1])
