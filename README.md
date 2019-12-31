# American Sign Language Detection

This repo implements a nueral network to perform a classification task on the [American Sign Language Finger Spelling Dataset](https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset). 

We implement 2 different models, both of them based on Convolutional Neural Networks. One of them works with the raw images, and the other one works with segmented images. 

To train your own model using our network, run:
```bash
python train.py
```

We do not have our saved model uploaded, so once you have trained your model, you can compute the test accuracy by running:
```bash
python test.py
```

Please check the paths to data, saved models etc inside the train.py and test.py files as this may change on your system. We have used absolute paths, and therefore, you may not be able to run this code directly on your set up. 

Please also read our final report if you would like know details on the project. 
