# ML_BPalgorithm_zzy
ML Final Task of Student Zhaoyu Zhang

The project uses BackPropagation to train neural network, and Perceptron algorithm to realise the classification, then an autoencoder is coded using BP algorithm. All the Machine Learning scripts are base on python, the packages needed are numpy, pandas and matplotlib. The project is coded using python-3.6 version, pycharm interpreter.

The author use Iris datasets to train backpropagation neural network algorithm,perceptron algorithm and autoencoder to finish the ML final 
task. In BackPropagationIris.py the basic neural network(4-8-3 netwok construction) is given, then the author uses PerceptronTrain.py to 
compare the performance of Perceptron with that of BP algorithm. In MoreHiddenLayerBP.py script. the author uses a two-hidden-layer neural 
network to do classification of Iris datasets; while in ThreeHiddenLayerBPNN.py the author uses a three-hidden-layer neural network to do
the classification. Finally, the autoencoder is trained by the author in BPAutoencoder.py script.

It is robust that in scripts MoreHiddenLayerBP.py and ThreeHiddenLayerBPNN.py, you can change the parameter "ActivationFunc", which supports "relu", "sigmoid" and "tanh" string, in the function InitializeTwolayer on the top. you can also change the dimension of hidden layer in several scripts, which locates in the main process, above the training loop.

All the python files are coded independently by Student Zhaoyu Zhang, BUAA.
