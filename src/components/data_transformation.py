import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    trainos_data_path: str=os.path.join('artifacts',"train_os.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            outliers = ["hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]

            cat = ['gender', 'smoking_history']
            
            no_outliers = ['age']
            
            outliers_pipeline= Pipeline( steps=
                                        [
                                         ("rs", RobustScaler())])
            
            no_outliers_pipeline = Pipeline( steps=
                                        [
                                         ("ss", StandardScaler())])

            cat_pipeline = Pipeline( steps=
                                  [
                                   ('ohe', OneHotEncoder())])
            
            preprocessor = ColumnTransformer(
                [
                    ("outliers_pipeline", outliers_pipeline, outliers),
                    ("no_outliers_pipeline", no_outliers_pipeline, no_outliers),
                    ("cat_pipeline", cat_pipeline, cat)
                ]
            )



            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train = pd.read_csv(train_path)

            logging.info("Read train data")
            
            test = pd.read_csv(test_path)

            logging.info("Read test data")

            os.makedirs(os.path.dirname(self.data_transformation_config.trainos_data_path),exist_ok=True)

            logging.info ("directory made for df_os")

            x_train_transf = train.drop('diabetes',axis=1)

            logging.info("Dropped target column from the train set to make the input data frame for model training")

            y_train_transf = train['diabetes']

            logging.info("Target feature obtained for model training")

            x_test_transf = test.drop('diabetes', axis=1)

            logging.info("Dropped target column from the test set to make the input data frame for model testing")
        
            y_test_transf = test['diabetes']

            logging.info("Target feature obtained for model testing")

            # print ("y_train classes:", y_train_transf.value_counts())

            # print ("y_test classes:", y_test_transf.value_counts())

            preprocessor = self.get_data_transformer_object()
            
            logging.info("Preprocessing object obtained")

            x_train_transf_preprocessed = preprocessor.fit_transform(x_train_transf)

            logging.info("Preprocessor applied on x_train_transf")

            x_train_transf_preprocessed_df = pd.DataFrame(x_train_transf_preprocessed)

            logging.info('''x_train_transf dataframe formed for feature selection by VIF''')
            
            for i in range(len(x_train_transf_preprocessed_df.columns)):
                
                x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.rename(columns={x_train_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info('''x_train_transf dataframe columns renamed''')
            
            # print ("x_train_preprocessed head:", x_train_transf_preprocessed_df.head(5))
            
            over_samp = SMOTE(k_neighbors=1)
            
            logging.info ("oversampling initiated")
            
            x_train_os, y_train_os = over_samp.fit_resample (x_train_transf_preprocessed_df, y_train_transf)

            logging.info ("oversampling completed")

            vif = pd.DataFrame()
            
            vif['vif'] = [variance_inflation_factor(x_train_os, i) for i in range (x_train_os.shape[1])]

            logging.info("VIF iniitiated")
            
            vif['features'] = x_train_os.columns

            logging.info("VIF completed")

            print (vif)
            
            high_vif_columns = vif[vif['vif'] > 10]['features'].tolist()
            
            print ("columns with vif>10:", high_vif_columns)

            x_train_vif = x_train_os.drop (high_vif_columns, axis=1)

            logging.info ("columns with vif>10 dropped")

            print ("x_train shape after dropping high vif columns", x_train_vif.shape)

            x_test_transf_preprocessed = preprocessor.transform(x_test_transf)

            logging.info("Preprocessor applied on x_test_transf")

            x_test_transf_preprocessed_df = pd.DataFrame(x_test_transf_preprocessed)

            logging.info('''x_test_transf dataframe formed for feature selection vif''')
            
            for i in range(len(x_test_transf_preprocessed_df.columns)):
                
                x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.rename(columns={x_test_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info('''x_test_transf dataframe columns renamed''')

            x_test_vif = x_test_transf_preprocessed_df.drop (high_vif_columns, axis=1)

            logging.info('''high VIF columns dropped from x_test''')
            
            train_arr = np.c_[np.array(x_train_vif), np.array(y_train_os)]
            
            logging.info("Combined the input features and target feature of the train set as an array.")
            
            test_arr = np.c_[np.array(x_test_vif), np.array(y_test_transf)]
            
            logging.info("Combined the input features and target feature of the test set as an array.")
            
            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor)
            
            logging.info("Saved preprocessing object.")
            
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,)
        
        except Exception as e:
            raise CustomException(e, sys)