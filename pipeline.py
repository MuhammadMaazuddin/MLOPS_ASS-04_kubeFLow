import kfp
from kfp import dsl
from kfp import compiler
from src.pipeline_components import data_preprocessing, model_training

@dsl.pipeline(
    name='MLOps Assignment 4 Pipeline',
    description='A pipeline that preprocesses data and trains a model.'
)
def mlops_pipeline(
    input_data_path: str = '/tmp/data/city_A.csv', # Default path, assuming it's available or passed
    test_size: float = 0.2,
    n_estimators: int = 100
):
    # Since the input is a local file (or we want to treat it as an artifact), 
    # we can use an importer if it's a URI, or if it's a local path in the container, 
    # we might need to handle it differently.
    # However, for this assignment, let's assume we use dsl.importer to bring in the data 
    # if it was a URL. Since it's a local file, we might need to assume the user uploads it 
    # or it's in a volume.
    # But to satisfy the InputPath requirement, we'll use dsl.importer with a dummy URI 
    # or the path if we assume it's accessible.
    
    # NOTE: In a real KFP setup, local files aren't accessible unless mounted.
    # For Minikube, we often use Minio or a PVC.
    # Here, I'll use dsl.importer to simulate passing the data artifact.
    # If the user runs this locally, they might need to adjust the path.
    
    # Let's use a placeholder URI or the local path.
    # If we use dsl.importer, it creates an artifact.
    
    importer_task = dsl.importer(
        artifact_uri=input_data_path,
        artifact_class=dsl.Dataset,
        reimport=False
    )
    
    preprocess_task = data_preprocessing(
        input_data_path=importer_task.output,
        test_size=test_size
    )
    
    train_task = model_training(
        x_train_path=preprocess_task.outputs['x_train_path'],
        y_train_path=preprocess_task.outputs['y_train_path'],
        n_estimators=n_estimators
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path='pipeline.yaml'
    )
