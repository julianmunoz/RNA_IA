#!/usr/bin/env python
# -*- coding: utf-8 -*-

from TrainingDataSet import TrainingDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer


class RNAGenerator(object):

    def initialize_network(self, input_l, hidden_l, output_l):
        # 3 inputs, 10 hidden and a single output neuron
        net = buildNetwork(input_l, hidden_l, output_l, bias=True, hiddenclass=TanhLayer)

        # Here we have generated a dataset that supports 3 dimensional inputs and one dimensional targets
        ds = SupervisedDataSet(input_l, output_l)
        return net, ds

    def train_network(self, net, ds, iterate_elem_creation):
        print "Entrenando la red ...\n"
        # Create sample for our problem
        data_generator = TrainingDataSet()
        data_dict = data_generator.get_data_set()

        for k, v in data_dict.iteritems():
            ds.addSample(k, v)

        # Backpropagation Trainer
        trainer = BackpropTrainer(net, ds, momentum=0.4, learningrate=0.07, weightdecay=0.01)

        trainer.trainUntilConvergence(verbose=True, maxEpochs=100)

        #self.validate_training(data_dict, net)

    def validate_training(self, data_dict, net):
        i = 0
        matched = 0
        for k, v in data_dict.iteritems():
            i += 1
            value_from_net = net.activate(k)
            print "Value from net: " + str(value_from_net) + " expected: " + str(v)
            if (float("%.1f" % value_from_net) - v[0]) <= 0.0:
                matched += 1
        print matched
        print str(matched * 100 / i) + '%'
        print i

    def work_with_the_RNA_live(self, net):
        print "Ingrese valores para obtener una respuesta de la red\n"
        print "Ingrese horario: 0.1 a 0.3 ,mañana 0.4 a 0.6  tarde, 0.7 a 0.9  noche\n"
        time = raw_input()
        print "Ingrese fecha: 0.1 a 0.3  Dia de la semana,0.4 a 0.6  Fin de semana,0.7 a 0.9  Fecha especial " \
              "(Feriados, Navidad, Año nuevo, etc)\n"
        date = raw_input()
        print "Ingrese consumo: 0.1 a 0.3  Poco, 0.4 a 0.6  Normal, 0.7 a 0.9  Extremo\n"
        usage = raw_input()
        inserted_data = (time, date, usage)
        print "Se ingreso\n"
        print inserted_data
        print "Valor de la red:\n"
        print net.activate(inserted_data)
        print "Apriete una tecla para salir"
        raw_input()
        return


gen_RNA = RNAGenerator()
net, ds = gen_RNA.initialize_network(3, 10, 1)
gen_RNA.train_network(net, ds, 180)
gen_RNA.work_with_the_RNA_live(net)







