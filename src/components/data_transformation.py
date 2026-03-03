import sys, os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import customException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    
    def get_data_transformer_object(self):
        
        """
        This function is responsible for data transformation
        
        """
         
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            ) 
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # make OneHotEncoder return dense arrays so StandardScaler can center
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler())
                ]
            )
            
            
            logging.info(f"categorical columns: {categorical_columns} ")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
              
        except Exception as e:
            raise customException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read Train and Test data completed.")
            
            # log column lists to help diagnose missing-column issues
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info(f"Test columns: {test_df.columns.tolist()}")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            
            # drop only the target column; pandas drop already infers axis from 'columns' argument
            # verify presence of target column before drop to provide clearer error
            if target_column_name not in train_df.columns:
                raise customException(
                    f"target column '{target_column_name}' not found in train data columns: {train_df.columns.tolist()}",
                    sys,
                )
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            if target_column_name not in test_df.columns:
                raise customException(
                    f"target column '{target_column_name}' not found in test data columns: {test_df.columns.tolist()}",
                    sys,
                )
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )
            
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info(f"saved preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise customException(e,sys)
        
        
