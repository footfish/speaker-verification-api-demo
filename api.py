import numpy as np
from flask import Flask
from flask import render_template
from flask_restful import reqparse,Resource, Api
from sklearn import mixture
import pickle #persistance of gmm model
from os import path
from os import remove
from time import time

app = Flask(__name__)
api = Api(app)

mfcc_train_store="mfcc_model_data.csv"
mfcc_test_store="mfcc_test_data.csv"
pickle_store="gmm.pickle"
n_gmmcomponents=5

parser = reqparse.RequestParser()
parser.add_argument('mfcc', type=list, location='json')


# Provide api informatino
class ApiInfo(Resource):
    def get(self):
        return render_template('test.html')
#        return {'version': '0.1',<!doctype html>
#            'usage': '/train /test /score'
#            }

# Status: Deletes test and train store & GMM pickle. 
class Delete(Resource):
    def get(self):
        if path.exists(mfcc_train_store):
            remove(mfcc_train_store)
        if path.exists(mfcc_test_store):
            remove(mfcc_test_store)
        if path.exists(pickle_store):
            remove(pickle_store)
        reply = {
        'deleted' : True,    
        }
        return reply, 200

# Status: Returns length of test and train store. 
class Status(Resource):
    def get(self):
        if path.exists(mfcc_train_store):
            mfcc_train_len = len(np.loadtxt(mfcc_train_store, delimiter=','))
        else:
            mfcc_train_len = 0
        
        if path.exists(mfcc_test_store):
            mfcc_test_len = len(np.loadtxt(mfcc_test_store, delimiter=','))
        else:
            mfcc_test_len = 0

        reply = {
        'mfcc_train_store_length' : mfcc_train_len,
        'mfcc_test_store_length' : mfcc_test_len,
        }
        return reply, 200


# Training - train model 
class Train(Resource):
    def put(self):
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        #validate features 
        # todo/check         
        #load the old mcc's (if any) and append mfcc 
        if path.exists(mfcc_train_store):
            mfcc = np.append(np.loadtxt(mfcc_train_store, delimiter=','),mfcc,axis=0)
        #save   
        np.savetxt(mfcc_train_store, mfcc, delimiter=",")  #save training array
        #load the pickle file (should speed it up)
        # create model
        gmm = mixture.GaussianMixture(n_gmmcomponents, covariance_type='full')
        #prune mfcc file if too large 
        # todo/check 
        #fit (warm start should help)
        tTime=time()
        gmm.fit(mfcc)
        tTime=time()-tTime
        #save the picked file 
        pickle.dump(gmm, open(pickle_store, 'wb')) #write a file 
        reply = {
        'mfcc_train_store_length' : len(mfcc),
        'training_time' : tTime,
        }
        return reply, 200


#Stores 'test' mfcc's that can be used to calculate a threshold for scoring
#The mfcc data should NOT be the same as that used for training but shuold be from a similar data set. 
class Test(Resource):
    def put(self):
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        #validate features 
        # todo 
        #load the test mfcc's (if any) and append mfcc 
        if path.exists(mfcc_test_store):
            mfcc = np.append(np.loadtxt(mfcc_test_store, delimiter=','),mfcc,axis=0)
        #save   
        np.savetxt(mfcc_test_store, mfcc, delimiter=",")  #save training array
        # load model
        with open(pickle_store, 'rb') as f:
            gmm = pickle.load(f)
        #score 
        testResult = gmm.score_samples(mfcc)
        scoreThreshold=np.average(testResult)-np.std(testResult)
        #store score_thereshold in persistent storage 
        # ** todo **

        reply = {
        'average': np.average(testResult), 
        'deviation': np.std(testResult), 
        'min' : np.min(testResult),
        'max' : np.max(testResult),
        'score_threshold': scoreThreshold, 
        'result_reference' : round(100*len(testResult[testResult>scoreThreshold])/len(testResult)), # % of matched frames in test data set
        'mfcc_test_store_length' : len(mfcc),        
        }
        return reply, 200

class Score(Resource):
    def put(self):
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        #validate features 
        # ** todo **
        # load model
        if path.exists(pickle_store):
            with open(pickle_store, 'rb') as f:
                gmm = pickle.load(f)
            #test score
            if path.exists(mfcc_test_store):
                mfccTest = np.loadtxt(mfcc_test_store, delimiter=',')
            testResult = gmm.score_samples(mfccTest)
            scoreThreshold=np.average(testResult)-np.std(testResult) #Average test score - Std dev. 
            #scoreThreshold=np.average(testResult) #Average test score
            #score 
            result = gmm.score_samples(mfcc)
            reply = {
                'average': np.average(result), 
                'deviation': np.std(result), 
                'min' : np.min(result),
                'max' : np.max(result),
                'length' : len(result),
                'score' : len(result[result>scoreThreshold]),
                'result_reference' : round(100*len(testResult[testResult>scoreThreshold])/len(testResult)), # % of matched frames in test data set
            }
        else:            
            reply = {
            'average': 0, 
            'deviation': 0, 
            'min' : 0,
            'max' : 0,
            'length' : 0,
            'score' : 0,
            'result_reference' : 0,
        }
        return reply, 200

@app.route('/') #serve static demo page 
def index():
    api_url = "this"
    return render_template('index.html')

api.add_resource(ApiInfo, '/doc')
api.add_resource(Status, '/status')
api.add_resource(Delete, '/delete')
api.add_resource(Train, '/train')
api.add_resource(Test, '/test')
api.add_resource(Score, '/score')

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response


if __name__ == '__main__':
    app.run(debug=True)