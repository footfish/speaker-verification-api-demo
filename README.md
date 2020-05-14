Install flask restful 
`pip install flask-restful`

Quick start 
`python api.py`

**Train:** Train the model 

`curl -XPUT --header "Content-Type: application/json" --data @mfcc_train.json http://127.0.0.1:5000/train`

**Test:** Test the model with a simi
`curl -XPUT --header "Content-Type: application/json" --data @mfcc_test.json http://127.0.0.1:5000/test`

**Score:**
`curl -XPUT --header "Content-Type: application/json" --data @mfcc_train.json http://127.0.0.1:5000/score`


### References 
* [Flask restfull Quickstart](https://flask-restful.readthedocs.io/en/latest/quickstart.html)
* [Deploying a Machine Learning Model as a REST API](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166)
* [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
* [Deploying Flask](https://flask.palletsprojects.com/en/1.0.x/deploying/)


# speaker-verification-api-demo
