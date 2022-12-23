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
- mlflow.set_experiment("nyc-taxi-experiment")