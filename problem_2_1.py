

import csv
import sys
import random
import random
import numpy as np

import cPickle
import gzip

import numpy as np
import random



def vectorized_result(j):
    e = np.zeros((3, 1))
    e[int(j)] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Neural_Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

        # print self.weights[0].shape
        # print self.weights[1].shape




    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


    def cross_entropy(self,batch_size, output, expected_output):
        return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))


    def backprop(self, x, y):

    	nabla_b = [np.zeros(b.shape) for b in self.biases]
    	nabla_w = [np.zeros(w.shape) for w in self.weights]

      	activation = x
        activations = [x]
        zs = []

        # print "activation"
        # print activation
        #
        # print "activations"
        # print activations

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)

            # print "z"
            # print z

            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cross_entropy(len(self.weights),activations[-1], y) * \
            sigmoid_prime(zs[-1])


        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # print "before",delta.shape
        # print delta



        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta

            # print delta.shape
            # print activations[-l-1].shape
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())


        # print "nabla_b"
        # print nabla_b
        #
        # print "nabla_w"
        # print nabla_w

     	return (nabla_b, nabla_w)





    def update_mini_batch(self, mini_batch, eta):


        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # print self.weights[1][0]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)




            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def check_postition(self,data):

        for i,x in enumerate(data):
            if(x==1.0):
                return i+1


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        count=0

        predicted=[]
        original=[]
        for x,y in test_results:
             # print x
             predicted.append(x)

             t=self.check_postition(y)
             original.append(t)

            # # print t
            # if(x==t):
            #     count=count+1

        # print "original"
        # print original
        # print "predicted"
        # print predicted

        correct = 0
        for i in range(len(original)):
            if original[i] == predicted[i]:
                correct += 1

        return correct



    def SGD(self, training_data, epochs, mini_batch_size, eta):

        n = len(training_data)

        test_data=""


        print "epoch",epochs

        print "mini_batch_size",mini_batch_size

        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]

        for k in range(0,len(mini_batches)):

                if k==0:
                    test_data = mini_batches[0]
                    mini_batches.pop(0)
                else:
                    mini_batches.append(test_data)
                    test_data = mini_batches[k]
                    mini_batches.pop(k)


                flag = 0

                for j in xrange(epochs):
                    # print "epoch", j
                    if flag == 1:
                        break




                    for mini_batch in mini_batches:
                        # print "inside loop"
                        self.update_mini_batch(mini_batch, eta)

                        # print mini_batch[0]

                    if test_data:
                        # print "outside loop////////////////////////////////////"
                        val = self.evaluate(test_data)
                        # print "after validation"
                        #
                        percent = val / float(len(test_data)) * 100
                        print "Number of fold :",k,"epoch :",j,"percentage",percent

                        if (percent > 80):
                            flag = 1






            


if __name__=="__main__":

    file = open('dermatology.data', 'r')

    list_of_labels=[]
    list_of_dataset=[]

    list_of_data=[]

    for line in file:
        data=[]
        if "?" in line:
            continue
        line = line.split(",")
        if(len(line)>0):
            for val in line:

                if(val==" "):
                    continue
                else:
                    data.append(val)
                    # print(val)
                    # print(float(val))

        # print(line)
        list_of_data.append(data)



    # print(list_of_data)
    DataSetList=[]
    for row in list_of_data:
        
        data = []
        for ele in row:
            # print(ele)
            data.append(float(ele))

        DataSetList.append(data)



    list_of_labels=[]
    list_of_dataset=[]
    for row in DataSetList:
        if row[0]==0:
            continue
        else:
            temp = vectorized_result(row[0]-1)
            list_of_labels.append(temp)
            list_of_dataset.append(row[1:len(row)])





list_of_dataset = np.asarray(list_of_dataset)

list_of_labels = np.array(list_of_labels)




temp_dataset=[]

for data in list_of_dataset:
    x=np.array(data).reshape(34,1)
    # print x
    temp_dataset.append(x)


list_of_dataset=temp_dataset


training_data = zip(list_of_dataset,list_of_labels)


# print training_data[0]
network = Neural_Network([34,16,3])

print "Number of hidden layers 16"

batch_size = len(training_data)/5
network.SGD(training_data,5, batch_size, 0.0001)




















