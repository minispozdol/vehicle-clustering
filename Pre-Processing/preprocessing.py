import os
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
#loading dataset
PATH_ROOT = Path(os.getcwd())
PATH_DATA = os.path.join(PATH_ROOT.parent,'vehicles.csv')
PATH_TEST = os.path.join(PATH_ROOT.parent,'vehicle_classifications.csv')

Data = pd.read_csv(PATH_DATA, engine = 'python')
TestDataLab = pd.read_csv(PATH_TEST, engine = 'python')

#creating a new attribute to store verified data
Data["VerifiedVariant"] = float("NaN")
#matching stock numbers to verified data to record data for verification
for i in range(len(TestDataLab)):
    Data.loc[Data['stock_number'] == TestDataLab.loc[TestDataLab.index[i],'stock_number'], 
             'VerifiedVariant'] = TestDataLab.loc[TestDataLab.index[i],'description']
             
#Train/testing data
ActualData=Data.loc[Data['stock_number'].reindex(Data.index, fill_value=False).isin(TestDataLab['stock_number'].tolist())]
#Raw Data
RawData=Data.loc[~Data['stock_number'].reindex(Data.index, fill_value=False).isin(TestDataLab['stock_number'].tolist())]

ActualData= ActualData[ActualData['VerifiedVariant'].notna()]

#filtering the attributes to be used as input for the algorithms
ProcessCategories = ['eeca_make', 'eeca_model', 'eeca_transmission', 'eeca_fuel',
                     'eeca_engine_power', 'eeca_weight', 'eeca_doors', 'eeca_engine_cc',
'eeca_vehicle_type', 'eeca_pollutants_stars', 'eeca_co2stars', 'eeca_co2per_km',
'eeca_yearly_co2', 'eeca_fuel_stars', 'eeca_yearly_cost', 'eeca_fuel_consumption',
'eeca_driver_safety_stars', 'VerifiedVariant', 'eeca_drive']

PP_Data=ActualData[ProcessCategories]
#replacing null values with 2WD as it is the only option
PP_Data['eeca_drive'].fillna(value = '2WD', inplace = True)

#selecting manufracture and model to use
TrainProcessData=PP_Data.loc[(PP_Data['eeca_make']=='TOYOTA') & (PP_Data['eeca_model'] =='COROLLA')]
#removing variants that are too small in numbers
TrainProcessData.drop(TrainProcessData.index[TrainProcessData['VerifiedVariant'] == '2004–08 Corolla Waggon Petrol'],
                      inplace = True)
#splitting the dataset into TrainSet and TestSet with same proportion of variants.
TrainSet, TestSet = train_test_split(TrainProcessData, test_size = 0.2, train_size = 0.8, shuffle = True,
                                     stratify = TrainProcessData['VerifiedVariant'])
#exporting to csv to be used in algorithms
TrainSet.to_csv('ToyCorollaTrainSET.csv')
TestSet.to_csv('ToyCorollaTestSET.csv')