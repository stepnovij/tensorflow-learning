# Evaluation tensorflow

Simple educational repo to evaluate tensorflow library.

## Currently implemented

* mnist dataset
  * logistic regression with constant training batch
  * logistic regression with dynamic training batches
  * nn with one relu
  * spurious local minimum [goodfellow](https://arxiv.org/pdf/1412.6544) with two hidden relus 
  * simple convolutional nn (best score on test data: 97.3%) 
  * lenet-5 (best score on test data: 98%) 
* Large Text Compression Benchmark
  * skip-gram model
  * cbow

Implementing parts of official [tensowflow's examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)
