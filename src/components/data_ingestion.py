# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass
# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer
# from src.components.data_transformation import DataTransformation


# @dataclass
# class DataIngestionConfig:
#     train_data_path: str = os.path.join("artifacts", "train.csv")
#     test_data_path: str = os.path.join("artifacts", "test.csv")
#     raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         logging.info("Initiating data ingestion")
#         try:
#             df = pd.read_csv("notebook/Data/stud.csv")
#             logging.info("Read the dataset as dataframe")

#             # Ensure the directory exists
#             os.makedirs(
#                 os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
#             )

#             # Save the raw dataset
#             df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

#             logging.info("Train-test split initiated")
#             train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#             # Save train and test datasets
#             train_set.to_csv(
#                 self.ingestion_config.train_data_path, index=False, header=True
#             )
#             test_set.to_csv(
#                 self.ingestion_config.test_data_path, index=False, header=True
#             )

#             logging.info("Data ingestion completed")

#             # return (
#             #     self.ingestion_config.train_data_path,
#             #     self.ingestion_config.test_data_path,
#             #     self.ingestion_config.raw_data_path,
#             # )
#             return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path


#         except Exception as e:
#             logging.error(f"Error in data ingestion: {e}")
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()


#     data_transformation = DataTransformation()
#     train_arr, test_arr = data_transformation.initiate_data_transformation(
#         train_data, test_data
#     )
#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_arr, test_arr))


import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv("notebook/Data/stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    result = data_transformation.initiate_data_transformation(train_data, test_data)

    print("Returned from data_transformation:", result)
    print("Type of result:", type(result))
    print("Length of result:", len(result))  # Ensure it's exactly 2

    if len(result) == 2:
        train_arr, test_arr = result
    else:
        raise ValueError("DataTransformation should return exactly two values")

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
