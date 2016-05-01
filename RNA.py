import random
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer


class GenerateDataSet(object):

    def generate_gender(self):
        return float("%s" % random.randint(0, 1))

    def generate_skinny(self):
        return (self.generate_gender(), float("%.1f" % random.uniform(0.1, 0.9)),
               float("%.1f" % random.uniform(0.1, 0.3))), (float("%.1f" % random.uniform(0.3, 0.4)),)

    def generate_common(self):
        return (self.generate_gender(), float("%.1f" % random.uniform(0.1, 0.9)),
           float("%.1f" % random.uniform(0.4, 0.6))), (float("%.1f" % random.uniform(0.1, 0.2)),)

    def generate_obese(self):
        return (self.generate_gender(), float("%.1f" % random.uniform(0.1, 0.9)),
                float("%.1f" % random.uniform(0.7, 0.9))), (float("%.1f" % random.uniform(0.7, 0.9)),)


class RNAGenerator(object):

    def initialize_network(self, input_l, hidden_l, output_l):
        # 3 inputs, 10 hidden and a single output neuron
        net = buildNetwork(input_l, hidden_l, output_l, bias=True, hiddenclass=TanhLayer)

        # Here we have generated a dataset that supports 3 dimensional inputs and one dimensional targets
        ds = SupervisedDataSet(input_l, output_l)
        return net, ds

    def train_network(self, net, ds, iterate_elem_creation):
        # Create sample for our problem
        data_dict = {}
        data_generator = GenerateDataSet()

        for elem in range(1, iterate_elem_creation):
            skinny_info, skinny_result = data_generator.generate_skinny()
            data_dict[skinny_info] = skinny_result
            common_info, common_result = data_generator.generate_common()
            data_dict[common_info] = common_result
            obese_info, obese_result = data_generator.generate_obese()
            data_dict[obese_info] = obese_result

        print data_dict
        for k, v in data_dict.iteritems():
            ds.addSample(k, v)

        # Backpropagation Trainer
        trainer = BackpropTrainer(net, ds)
        trainer.trainUntilConvergence(verbose=True, maxEpochs=60)

        i = 0
        for k, v in data_dict.iteritems():
            i += 1
            print "Value from net: " + str(net.activate(k)) + " expected: " + str(v)
            if i == 50:
                break


gen_RNA = RNAGenerator()
net, ds = gen_RNA.initialize_network(3, 10, 1)
gen_RNA.train_network(net, ds, 180)


"""
Data set:

Sexo:  Hombre=0; Mujer=1

0.1 a 0.3 (Adolescente) edad entre 10 y 19 anos.
0.4 a 0.7 (Joven Adulto) edad entre 20 y 30.
0.8 a 0.9 (Adulto) edad entre 30 en adelante.

0.1 a 0.3  flaco
0.4 a 0.6 normal
0.7 a 0.9 obeso

0.1 big mac triple mac
0.2 doble cuarto de libra cuarto de libra con queso
0.3 magnifica
0.4 hamburguesa con queso
0.5 mc fiesta
0.6 hamburguesa
0.7 angus deluxe
0.8 angus bacon
0.9 angus tasty

"""





