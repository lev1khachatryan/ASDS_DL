# NeuralNetwork
Fully Connected Neural Network implementation using tensorflow.

The idea is to make it easy to construct, train and save a
 neural network with several lines of code. 
The code is written in such a way that the neural network 
object is similar to sklearn's other ML models. To have a better
intuition on how to use the neural networks feel free to look into
test_autoencoder and test_classificator jupyter notebooks.

If you will be using docker and you have access to GPU, uncomment the first line in the Dockerfile and comment the second one.
 
If you won't be using docker:

If you have access to GPU replace `tensorflow==1.12` with `tensorflow-gpu==1.12` in the `requirements.txt` file.

To install needed libraries just run
```
pip install -r requirements.txt
```