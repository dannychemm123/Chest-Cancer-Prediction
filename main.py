from cnnClassifier import logger
from cnnClassifier.pipeline.data_ingestion_stage_01 import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.prepare_base_model_stage_02 import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.model_trainer_stage_03 import ModelTrainingPipeline
from cnnClassifier.pipeline.model_evaluation_stage_04 import ModelEvaluationPipeline

STAGE_NAME = "DATA INGESTION STAGE"



try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Started Successfully')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Completed Successfully\n\nx=======x')
except Exception as e:
    logger.exception(e)
    
STAGE_NAME = 'Prepare base stage'

try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Started Successfully')
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Completed Successfully\n\nx=======x')
except Exception as e:
    logger.exception(e)
    
STAGE_NAME = 'Training'
try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Started Successfully')
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Completed Successfully\n\nx=======x')
except Exception as e:
    logger.exception(e)
    
    
STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Started Successfully')
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME}<<<<<<<< Completed Successfully\n\nx=======x')
except Exception as e:
    logger.exception(e)
    