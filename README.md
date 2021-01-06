# weatherclassification
MLP Weather Classification Neural Network

Data: https://www.kaggle.com/budincsevity/szeged-weather

This model was built to preempt, based on a simple table of given weather conditions, whether the day would be rainy or snowy. It demonstrates the efficiency of deep learning on large datasets with many recordings. The benefit of a model like this is that this classification required no data that a sensor-equipped small weather station can’t provide.

The model itself is a multi-layer perceptron with 227,219 trainable parameters. It has 5 dense layers: the first four utilize the ReLU and Sigmoid activation functions and the fifth layer consists of only 3 units and utilizes a Softmax activation in order to classify rain or snow.

The model achieves a prediction accuracy of just over 99% using dense layers and an MLP architecture.
