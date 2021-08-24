from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


class Model_Finder:
    """    
            In this class we have to find the best model which are better accuracy score and 
            better roc auc score we have to try all the model which are used and select those
            model which has better accracy score
    """

    def __init__(self,file_object,logger_object):
        self.file_object          =        file_object
        self.logger_object        =        logger_object
        self.gnb                  =        GaussianNB()
        self.xgb                  =        XGBClassifier(objective='binary:logistic', n_jobs=-1)


    def get_best_params_for_naive_bayes(self,train_x,train_y):
        """   
            Method Name: get_best_params_for_naive_bayes
            Description: get the parameter for Naive Bayes's Algorithm which gives best accuracy
            Output     : The model with best parameter
            On Failure : raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing the different combination of parameter
            self.param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}

            # creating an object of the grid search
            self.grid = GridSearchCV(estimator=self.gnb, param_grid=self.param_grid, cv=5, verbose=3)

            # finding the best parameter
            self.grid.fit(train_x,train_y)

            # extracting the best parameter
            self.var_smoothing = self.grid.best_params_["var_smoothing"]

            #creating a new model with best parameter
            self.gnb   = GaussianNB(var_smoothing=self.var_smoothing)

            # training the mew model
            self.gnb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):
        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception       
        """

        self.logger_object.log(self,file_object,
                    'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        
        try:
            # initializing the different combination of parameter
            self.param_grid_xgboost = {
                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(8, 10, 1)
            }
            # Creating an object of the Grid Search class
            self.grid  =  GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3, cv=5)
            self.grid.fit(train_x,train_y)

            # Extracting the best parameter
            self.criterion       =     self.grid.best_params_['criterion']
            self.max_depth       =     self.grid.best_params_['max_depth']
            self.n_estimators    =     self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(criterion=self.criterion, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1)

            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


