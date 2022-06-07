{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/riyasoy/MachineLeearningprojects/blob/main/Handwritten_Digit_Recognition_with_Tensorflow.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "id": "f31be3c0",
      "metadata": {
        "id": "f31be3c0"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import cv2 \n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import tensorflow as tf"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "id": "71a9f272",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "71a9f272",
        "outputId": "9e2bf609-ba79-40e0-ab61-9fed429d7222"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " WELCOME TO RIYA'S HANDWRITTEN DIGITS RECOGNITON SYSTEM\n",
            "Epoch 1/3\n",
            "1875/1875 [==============================] - 5s 2ms/step - loss: 0.2642 - accuracy: 0.9232\n",
            "Epoch 2/3\n",
            "1875/1875 [==============================] - 4s 2ms/step - loss: 0.1062 - accuracy: 0.9671\n",
            "Epoch 3/3\n",
            "1875/1875 [==============================] - 4s 2ms/step - loss: 0.0727 - accuracy: 0.9766\n",
            "313/313 [==============================] - 1s 1ms/step - loss: 0.0977 - accuracy: 0.9684\n",
            "0.09767451882362366\n",
            "0.9684000015258789\n",
            "INFO:tensorflow:Assets written to: digits.model/assets\n"
          ]
        }
      ],
      "source": [
        "\n",
        "\n",
        "print (\" WELCOME TO RIYA'S HANDWRITTEN DIGITS RECOGNITON SYSTEM\")\n",
        "#decide if to load an existing model or to train a new one\n",
        "train_new_model = True\n",
        "\n",
        "if train_new_model :\n",
        "    #load the dataset\n",
        "    mnist = tf.keras.datasets.mnist\n",
        "    (X_train, y_train),(X_test,y_test) = mnist.load_data()\n",
        "    #normalizing the data  \n",
        "    X_train = tf.keras.utils.normalize(X_train, axis = 1)\n",
        "    X_test = tf.keras.utils.normalize(X_test, axis = 1)\n",
        "\n",
        "    # Create a neural network model\n",
        "    # Add one flattened input layer for the pixels\n",
        "    # Add two dense hidden layers\n",
        "    # Add one dense output layer for the 10 digits\n",
        "\n",
        "\n",
        "    model = tf.keras.models.Sequential()  \n",
        "    #now adding some layers, specified input shape\n",
        "    #these are pixels shape of all individual images and we feed them into\n",
        "    #the input layers\n",
        "    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))\n",
        "    model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))\n",
        "    model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))\n",
        "    #we will have output layer which will be a dense layer\n",
        "    model.add(tf.keras.layers.Dense(units=10, activation = tf.nn.softmax))\n",
        "\n",
        "    # Compiling and optimizing model\n",
        "    model.compile(optimizer='adam',loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "\n",
        "\n",
        "    #fit the model\n",
        "    model.fit(X_train, y_train, epochs = 3)\n",
        "    #epochs tells how many times we are going to see the data, \n",
        "    #or how many times we are going to process\n",
        "\n",
        "    # evaluate the model\n",
        "    val_loss, val_acc= model.evaluate(X_test, y_test)\n",
        "    print(val_loss)\n",
        "    print(val_acc)\n",
        "    #saving the model\n",
        "    model.save('digits.model') #to scan the images\n",
        "\n",
        "else:\n",
        "    #load the model\n",
        "    model = tf.keras.models.load_model('handwritten_digits.model')\n",
        "\n",
        "#Load custom images and predict them\n",
        "image_number = 1\n",
        "while os.path.isfile('digits/digit{}.png'.format(image_number)):\n",
        "    try:\n",
        "        img = cv2.imread('/2.png'.format(image_number))[:,:,0]\n",
        "        img = np.invert(np.array([img]))\n",
        "        prediction = model.predict(img)\n",
        "        print(\"The number is probably a {}\".format(np.argmax(prediction)))\n",
        "        plt.imshow(img[0], cmap=plt.cm.binary) #to see how it looks like\n",
        "        plt.show()\n",
        "        image_number +=1\n",
        "    except:\n",
        "        print(\"Error reading image! Proceeding with next image...\")\n",
        "        image_number +=1"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "-vdSwmjBkHBr"
      },
      "id": "-vdSwmjBkHBr",
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.8"
    },
    "colab": {
      "name": "Handwritten Digit Recognition with Tensorflow.ipynb",
      "provenance": [],
      "toc_visible": true,
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}