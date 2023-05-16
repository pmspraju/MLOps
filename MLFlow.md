# MLFlow Commands: https://mlflow.org/ 

## Maturity Model

### Level 0 - Devops - No, MLops - No

 - No Automation
 - Code in Jupyter notebooks
 
### Level 1 - Devops - Yes, MLops - No

 - Releases are automated
 - Unit and integration test cases
 - CI/CD
 - Ops metrics
 - No ML experimenting tracking
 - No reproducibility
 - Data science separated from Engineering
 
### Level 2 - Automated Training, Devops - Yes, MLops - Yes

 - Traning pipeline
 - Experiment tracking
 - Model registry
 - Low friction deployement
 - Datascience working with Engineering 
 
### Level 3 - Automated Deployement 
 
 - Easy to deploye model
 - Exists a ML platform to easy deploy
 - A/B test - to determine better version in a given model
 - Model monitoring 
 
### Level - 4 - Fully Automated 

 - All above features plus 
 - No/minimal manual intervention
 - Automatic running of pipeline and dertermine better model to deploy
   in caseof failures. 

### Components of MLFlow

 <br> a. Tracking 
 <br> b. Models 
 <br> c. Model Registry
 <br> d. Projects 
 
 
### Experiment Tracking 

- Relevant information - 
 <br> a. Source code
 <br> b. Environment
 <br> c. Data
 <br> d. Model
 <br> e. Hyperparameters
 <br> f. Metrics
 
- Start MLflow - using a backend sqlite to store artifacts. 
- Goto the folder where mlflow.db exists (home/pmspraju/tracking-server/), issue below command 
<br>  mlflow ui --backend-store-uri sqlite:///mlflow.db


- mlflow.set_tracking_uri("sqlite:////home/pmspraju/tracking-server/mlflow.db")
- Create an experiment - if experiment name is not unique, exception is given

```
experiment_id = mlflow.create_experiment(
        "nyc-taxi-experiment-1",
        #artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        artifact_location='//home/pmspraju/tracking-server/mlruns/',
        tags={"version": "v1", "priority": "P1"},
)
```

- mlflow.set_experiment("nyc-taxi-experiment")

- Start the run of the experiment and track

```
with mlflow.start_run():
  mlflow.set_tag("developer", "Madhu")
  mlflow.log_param("train-data-path", "/home/pmspraju/tracking-server/data/green_tripdata_2021-01.parquet")
  mlflow.log_param("valid-data-path", "/home/pmspraju/tracking-server/data/green_tripdata_2021-02.parquet")

  alpha = 0.1
  mlflow.log_param("alpha", alpha)
  
  mlflow.set_tag("model", "lasso")
  lr = Lasso(alpha)
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_val)
  rmse = mean_squared_error(y_val, y_pred, squared=False)
  mlflow.log_metric("rmse", rmse)

  mlflow.log_artifact(local_path="/home/pmspraju/tracking-server/models/lin_reg.bin", artifact_path="models_pickle")
```

- Command to run mlflow in aws console 

```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://db_user_name:db_password@aws_db_endpoint:5432/db_name --default-artifact-root s3://s3_bucket_name
```
