# Philosophy Text Generator from Scratch

I created a text generator using a single book, [The Critique of Pure Reason](https://www.gutenberg.org/ebooks/4280) by Immanuel Kant, and a small neural network model trained for only a few epochs on short segments of the text. 

## Features
- a training loop created from scratch, implemented with a container class and many methods
- datasets, batch creation, tokenization, and numericalization from scratch using my own classes and methods
- creation of my own training samples from a raw source (a complete book) instead of a curated dataset
- a model created from the basic modules of PyTorch, implementing weight tying
- a custom text generation method, using an altered sotmax function with a customizable exponential base to tune the randomness of token selection
- a custom loss function and metric function, which ignore out-of-vocabulary tokens when evaluating the model's success

The main goal of this project was to build a model and training system from the most basic building blocks of PyTorch possible, without even using its built-in data loading functionality. 

Although the text generation actually works pretty well with such a small model and short training period, the bulk of the work in this project was the surrounding infrastructure to allow for the training process to happen at all. I learned a lot about the lower level details of the full process of starting with a raw data source and ending up with a trained model that can do something useful.

## Running this project on your machine

This project has three main jupyter notebooks, which need to be run in order. They will download the data (the book), train a model, and then generate some text. Random seeds are used throughout, so the results should be completely reproducible. The only requirements are PyTorch, NumPy, and Pandas.
