# Speaker Verification API demo
This is a demonstration of speaker verification (aka speaker authentication) using a REST API.

_This was part of an MSC desertation. [Read full paper](./dissertation/A_REST_API_approach_to_Distributed_Speaker_Verification.pdf)_

## Summary:


Normally speaker verification (authentication) systems are centralised. Meaning audio is recorded, then sent to the centralised system for processing. That's inefficent for Web applications. This demonstrates an alternative distributed approach. Speech 'features' are extracted in the browser, then sent via REST API for authentication. 

**Note**: _This is an academic exercise, not production code._

![Traditional approach vs this approach](./images/oldvnew.png)

## How to use it:
You will need to perform the following steps; 

1. ***Training*** - This 'trains' the model with your speech and sets a benchmark for your speech. . 
2. ***Scoring*** - This is the 'verification', the higher the score the more likely it's you (if it works!). Obviously you will need to have trained and tested the model before scoring will work. 

## How it works:
Your browser will use the microphone to capture spoken digits, then process these to feature vectors ([MFCC's - mel frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)). The feature vectors are sent to a remote API for speaker verification. The browser uses [Web Audio API ](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) and [Meyda JS library](https://meyda.js.org/guides/online-web-audio) to capture the spoken data. The remote API uses a GMM ([Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model)) machine learning method for speaker verification.


## Screenshots
Training the model:
![Training screenshot](./images/training.png)

## Demonstration 
See a (very exciting) [youtube video](https://youtu.be/Hd9TVjA0NCE)


## Installation:
Install python from <https://www.python.org/>
Install Mongodb locally <https://www.mongodb.com/> or use a hosted service like <https://www.mlab.com/> 

### Clone 

```
git clone https://github.com/footfish/speaker-verification-api-demo.git
cd speaker-verification-api-demo
```

### (Optionally) Setup Python virtual env. 
Python 3.3 or newer, the venv module is included in the Python standard library and requires no additional installation
```
#Linux/Mac 
python3 -m venv env
source env/bin/activate
which python #verify using virtual env

#Windows
py -m venv env
.\env\Scripts\activate #bash users try 'source env/Scripts/activate'
where python #verify using virtual env
```

### Install Required Libs 
`pip install -r requirements.txt`

### Set required config variables 
```
#Linux/Mac
export JWT_SECRET_KEY='your-secret-here'
export MONGODB_URI='mongodb://your-user:your-pass@your-host:your-port/your-dbname'

#Windows
set JWT_SECRET_KEY='your-secret-here'
set MONGODB_URI='mongodb://your-user:your-pass@your-host:your-port/your-dbname'
```

### Quick start 
`python api.py`

## Heroku Deployment:
The application can quickly be deployed to [Heroku](https://www.heroku.com/). 
You will need a Heroku account and the [Heruku cli](https://devcenter.heroku.com/articles/heroku-cli) installed. 
_This deployment used Heroku mongolab addon which was discontinued on November 10, 2020, you will need to use an equivalent MongoDb service._
```
heroku create
#Env. variables (If using mongolab MONGODB_URI will be set up automatically)
heroku config:set JWT_SECRET_KEY='your-secret-here'
heroku config
git push heroku master
heroku open
```

#### Sample data 
Optionally you can load the provided [sample database](../samples/sample_db/userdata.bson) of 35 users. This is useful for benchmarking and intrusion testing.

### Background 
You will find a good step-by-step example of Python speech processing code [here](https://github.com/footfish/python-speechprocessing-example) 

### References 

#### Speech Processing/Machine Learning 
* [Cepstrum and MFCC - Introduction to Speech Processing - Aalto University Wiki ](https://wiki.aalto.fi/display/ITSP/Cepstrum+and+MFCC)
* [Deploying a Machine Learning Model as a REST API](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166)
* [Practical Cryptography - Guide MFCC's](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
* [Gaussian Mixture Modelling (GMM) - Towards Data Science ](https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f)
* [Machine Learning with Python: Expectation Maximization and Gaussian Mixture Models in Python ](https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php)


#### API/Lib 
* [Web Audio API](https://www.w3.org/TR/webaudio/)
* [Scikit-learn - Machine Learning in Python](https://scikit-learn.org/)
* [python_speech_features library](https://github.com/jameslyons/python_speech_features)
* [Meyda JSDoc: Class: MeydaAnalyzer](https://meyda.js.org/reference/MeydaAnalyzer.html)
* [Meyda JSDoc: Home](https://meyda.js.org/reference/index.html)
* [Meyda JSDoc: Source: main.js](https://meyda.js.org/reference/main.js.html)
* [librosa/librosa: Python library for audio and music analysis](https://github.com/librosa/librosa)
* [Meyda Features](https://meyda.js.org/audio-features)
* [Yaafe core features — YAAFE - Yet Another Audio Feature Extractor](http://yaafe.sourceforge.net/features.html)
* [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
* [Flask-JWT-Extended’s Documentation](https://flask-jwt-extended.readthedocs.io/en/stable/)
* [PyMongo Documentation](https://pymongo.readthedocs.io/en/stable/)


#### Other, Python/Flask/Heroku
* [Flask restfull Quickstart](https://flask-restful.readthedocs.io/en/latest/quickstart.html)
* [Deploying Flask](https://flask.palletsprojects.com/en/1.0.x/deploying/)
* [Getting Started on Heroku with Python](https://devcenter.heroku.com/articles/getting-started-with-python#prepare-the-app)
* [How to create a REST API in Python using the Flask Microframework](https://medium.com/the-andela-way/how-i-developed-an-api-in-python-using-flask-4e388674f1)
* [Welcome to Flask — Flask Documentation (1.1.x)](https://flask.palletsprojects.com/en/1.1.x/)
* [Templates — Flask Documentation (1.1.x) ](https://flask.palletsprojects.com/en/1.1.x/tutorial/templates/)
* [API — Flask Documentation (1.1.x) ](https://flask.palletsprojects.com/en/1.1.x/api/#flask.g)
* [Getting Started With Flask, A Python Microframework ― Scotch.io ](https://scotch.io/tutorials/getting-started-with-flask-a-python-microframework)
* [Serving Static Files with Flask ](https://stackabuse.com/serving-static-files-with-flask/)
* [Getting Started on Heroku with Python | Heroku Dev Center ](https://devcenter.heroku.com/articles/getting-started-with-python#define-a-procfile)
* [Heroku Python Support | Heroku Dev Center ](https://devcenter.heroku.com/articles/python-support)
* [heroku/python-getting-started: Getting Started with Python on Heroku. ](https://github.com/heroku/python-getting-started)
* [Deploying with Git | Heroku Dev Center ](https://devcenter.heroku.com/articles/git)
* [Creating Web APIs with Python and Flask | Programming Historian ](https://programminghistorian.org/en/lessons/creating-apis-with-python-and-flask)
* [Quickstart — Flask-RESTful 0.3.8 documentation ](https://flask-restful.readthedocs.io/en/latest/quickstart.html)
* [JWT authorization in Flask - codeburst](https://codeburst.io/jwt-authorization-in-flask-c63c1acf4eeb)


