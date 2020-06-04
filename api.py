import config as cfg
import numpy as np
from flask import Flask, abort, render_template
from flask_restful import reqparse,Resource, Api
from flask_jwt_extended import create_access_token,JWTManager,jwt_required, get_jwt_identity
import datetime
from sklearn import mixture
import pickle #persistance of gmm model
from os import path
from os import remove
from time import time
from pymongo import MongoClient
from bson import Binary

dbClient = MongoClient(cfg.MONGODB_URI,retryWrites=False)  
userData = dbClient[dbClient.get_default_database().name].userdata 

app = Flask(__name__)
api = Api(app)
app.config['JWT_SECRET_KEY'] = cfg.JWT_SECRET_KEY
jwt = JWTManager(app) #initialise JWT 

n_gmmcomponents=5
discard_mfcc0=True #choose to discard first cepstral coefficient 

parser = reqparse.RequestParser()
parser.add_argument('mfcc', type=list, location='json')
parser.add_argument('username')

# Provide api information
class ApiInfo(Resource):
    def get(self):
        return {'version': '0.1',
                'usage': '/doc /login /status /train /test /score /delete'
                }

# 'Login user' and return JWT
class Login(Resource):
  def post(self):
    args = parser.parse_args()
    user = userData.find_one({"email": args['username']})
    if user != None:
        print(args['username'] + ' logged in')
        expires = datetime.timedelta(days=7)
        access_token = create_access_token(identity=user['userId'], expires_delta=expires)
        return {'token': access_token}, 200
    else:
        return {'error': 'Email invalid'}, 401

# Status: Deletes test and train store & GMM pickle. 
class Delete(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()
        userData.update_one({'userId':JWT_userId},{'$unset': {'mfcc-train-store': ""}})
        userData.update_one({'userId':JWT_userId},{'$unset': {'mfcc-test-store': ""}})
        userData.update_one({'userId':JWT_userId},{'$unset': {'gmm-pickle-store': ""}})
        reply = {
        'deleted' : True,    
        }
        return reply, 200

# Status: Returns length of test and train store. 
class Status(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0,'mfcc-train-store':1,'mfcc-test-store':1 })
        if db_result != None: 
            if 'mfcc-train-store' in db_result:
                mfcc_train_len=len(pickle.loads(db_result['mfcc-train-store']))
            else:
                mfcc_train_len = 0
            if 'mfcc-test-store' in db_result:
                mfcc_test_len=len(pickle.loads(db_result['mfcc-test-store']))
            else: 
                mfcc_test_len = 0
            reply = {
            'mfcc_train_store_length' : mfcc_train_len,
            'mfcc_test_store_length' : mfcc_test_len,
            }
            return reply, 200
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500


# Training - train model 
class Train(Resource):
    @jwt_required    
    def put(self):
        JWT_userId = get_jwt_identity()
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])     
        if (discard_mfcc0):
            mfcc = np.delete(mfcc, 0, axis=1) #delete the 1st mfcc coeffecient 
        #validate features 
        # todo/check         

        #load the old mcc's (if any) and append mfcc 
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0,'mfcc-train-store':1})
        if db_result != None: 
            if 'mfcc-train-store' in db_result:
                mfcc=np.append(pickle.loads(db_result['mfcc-train-store']),mfcc,axis=0)
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500

        #save training store 
        userData.update_one({'userId':JWT_userId},{'$set': {'mfcc-train-store': Binary(pickle.dumps(mfcc))}})

        #load the pickle file (should speed it up)
        # create model
        gmm = mixture.GaussianMixture(n_gmmcomponents, covariance_type='full')
        #prune mfcc file if too large 
        # todo/check 
        #fit (warm start should help)
        tTime=time()
        gmm.fit(mfcc)
        tTime=time()-tTime
        #save gmm pickle 
        userData.update_one({'userId':JWT_userId},{'$set': {'gmm-pickle-store': Binary(pickle.dumps(gmm))}})

        reply = {
        'mfcc_train_store_length' : len(mfcc),
        'training_time' : tTime,
        }
        return reply, 200

#Stores 'test' mfcc's that can be used to calculate a threshold for scoring
#The mfcc data should NOT be the same as that used for training but shuold be from a similar data set. 
class Test(Resource):
    @jwt_required        
    def put(self):
        JWT_userId = get_jwt_identity()
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        if (discard_mfcc0):
            mfcc = np.delete(mfcc, 0, axis=1) #delete the 1st mfcc coeffecient 
        #validate features 
        # ** TODO **

        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0,'mfcc-test-store':1,'gmm-pickle-store':1})
        if db_result != None: 
            #load the test mfcc's (if any) and append mfcc 
            if 'mfcc-test-store' in db_result:
                mfcc=np.append(pickle.loads(db_result['mfcc-test-store']),mfcc,axis=0)
            #load GMM model
            if 'gmm-pickle-store' in db_result:
                gmm=pickle.loads(db_result['gmm-pickle-store'])
            else: 
                return {'error': 'this is unexpected, no trained model found'}, 500
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500

        #save test store 
        userData.update_one({'userId':JWT_userId},{'$set': {'mfcc-test-store': Binary(pickle.dumps(mfcc))}})

        #score 
        testResult = gmm.score_samples(mfcc)
        scoreThreshold=np.average(testResult)-np.std(testResult)

        #store score_thereshold in persistent storage 
        # ** TODO **

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
    @jwt_required        
    def put(self):
        JWT_userId = get_jwt_identity()        
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        if (discard_mfcc0):
            mfcc = np.delete(mfcc, 0, axis=1) #delete the 1st mfcc coeffecient 
        #validate features 
        # ** TODO **

        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0,'mfcc-test-store':1,'gmm-pickle-store':1})
        if db_result != None: 
            #load the test mfcc's 
            if 'mfcc-test-store' in db_result:
                mfccTest=pickle.loads(db_result['mfcc-test-store'])
            # load GMM model
            if 'gmm-pickle-store' in db_result:
                gmm=pickle.loads(db_result['gmm-pickle-store'])
               #test score - # NOTE would be better to store in db and load
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
            else: #no model trained 
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
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500

#serve static html templates
@app.route('/') 
def index():
    return render_template('index.html')
@app.route('/session') 
def session():
    return render_template('login.html')

#add API resources 
api.add_resource(ApiInfo, '/doc')
api.add_resource(Status, '/status')
api.add_resource(Delete, '/delete')
api.add_resource(Train, '/train')
api.add_resource(Test, '/test')
api.add_resource(Score, '/score')
api.add_resource(Login, '/login')

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

if __name__ == '__main__':
    app.run(debug=True)

