from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "DATA INGESTION STAGE"

class DataIngestionTrainingPipeline:
    ''' Training pipeline for training data'''
    def __init__(self):
        pass
    
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data_file()
        data_ingestion.extract_zip_file()
    


if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Started Successfully')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Completed Successfully\n\nx=======x')
    except Exception as e:
        logger.exception(e)
        
        
    