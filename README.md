# Recommendations

An end-to-end anime recommendation system in Google Cloud Platform. It will automatically retrain new data, redeploy follow schedule by using Google Cloud Composer. The data set contains information on user preference data from 73,516 users on 12,294 anime. 
[Demo recommendation system](https://dautroc.shinyapps.io/recommendation/)
[Anime dataset](https://www.kaggle.com/CooperUnion/anime-recommendations-database)


## Before you begin

1. Create a new [Cloud Platform project](https://console.cloud.google.com/projectcreate).

2. [Enable billing](https://support.google.com/cloud/answer/6293499#enable-billing)
   for your project.

3. [Enable APIs](https://console.cloud.google.com/apis/dashboard) for
  * BigQuery API
  * Cloud Resource Manager
  * AI Platform Training & Prediction API
  * App Engine Admin
  * Container Engine (if using Airflow on GKE)
  * Cloud SQL API    (if using Airflow on GKE)
  * Cloud Composer API (if using Cloud Composer for Airflow)

## Data

![plot](./Recommendation-system/image/rating1.png)
-1 rating don't have many value so i remove it and create the new dataset.


## Model
Model code is contained in the wals_ml_engine directory.
wals.py— creates the WALS model; executes the WALS algorithm; calculates the root-mean-square error (RMSE) for a set of row/column factors and a ratings matrix.
model.py — loads the dataset; creates two sparse matrices from the data, one for training and one for testing; executes WALS on the training sparse matrix of ratings.

# Preprocess data

The item mapping is accomplished using the following [numpy](http://www.numpy.org/) code. The code creates an array of size [0..max_item_id] to perform the mapping, so if the maximum item ID is very large, this method might use too much memory.


np_items = ratings_df.item_id.as_matrix()
unique_items = np.unique(np_items)
n_items = unique_items.shape[0]
max_item = unique_items[-1]

map unique items down to an array 0..n_items-1
z = np.zeros(max_item+1, dtype=int)
z[unique_items] = np.arange(n_items)
i_r = z[np_items]
The code for mapping users is essentially the same as the code for items.

The model code randomly selects a test set of ratings. By default, 10% of the ratings are chosen for the test set. These ratings are removed from the training set and will be used to evaluate the predictive accuracy of the user and item factors.


test_set_size = len(ratings) / TEST_SET_RATIO
test_set_idx = np.random.choice(xrange(len(ratings)),
                                size=test_set_size, replace=False)
test_set_idx = sorted(test_set_idx)

ts_ratings = ratings[test_set_idx]
tr_ratings = np.delete(ratings, test_set_idx, axis=0)
Finally, the code creates a scipy sparse matrix in coordinate form (coo_matrix) that includes the user and item indexes and ratings. The coo_matrix object acts as a wrapper for a sparse matrix. It also performs validation of the user and ratings indexes, checking for errors in preprocessing.

tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))

# WAL Algorithm
After the data is preprocessed, the code passes the sparse training matrix into the TensorFlow WALS model to be factorized into row factor X and column factor Y.

The TensorFlow code that executes the model is actually simple, because it relies on the WALSModel class included in the contrib.factorization_ops module of TensorFlow.

A SparseTensor object is initialized with user IDs and items IDs as indices, and with the ratings as values. From wals.py:


input_tensor = tf.SparseTensor(indices=zip(data.row, data.col),
                                values=(data.data).astype(np.float32),
                                dense_shape=data.shape)
The data variable is the coo_matrix object of training ratings created in the preprocessing step.

The model is instantiated:


model = factorization_ops.WALSModel(num_rows, num_cols, dim,
                                    unobserved_weight=unobs,
                                    regularization=reg,
                                    row_weights=row_wts,
                                    col_weights=col_wts)
The row factors and column factor tensors are created automatically by the WALSModel class, and are retrieved so they can be evaluated after factoring the matrix:


# retrieve the row and column factors
row_factor = model.row_factors[0]
col_factor = model.col_factors[0]
The training process executes the following loop within a TensorFlow session using the simple_train method in wals.py:


row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

sess.run(model.initialize_op)
sess.run(model.worker_init)
for _ in xrange(num_iterations):
    sess.run(model.row_update_prep_gramian_op)
    sess.run(model.initialize_row_update_op)
    sess.run(row_update_op)
    sess.run(model.col_update_prep_gramian_op)
    sess.run(model.initialize_col_update_op)
    sess.run(col_update_op)
After num_iterations iterations have been executed, the row and column factor tensors are evaluated in the session to produce numpy arrays for each factor:


# evaluate output factor matrices
output_row = row_factor.eval(session=session)
output_col = col_factor.eval(session=session)
These factor arrays are used to calculate the RMSE on the test set of ratings. The two arrays are also saved in the output directory in numpy format.



## Installation

### Install code
In the new terminal window, update your instance's software respositories.


sudo apt-get update
Install git, bzip2, and unzip.


sudo apt-get install -y git bzip2 unzip
Clone the sample code repository:


git clone https://github.com/dautroc1/Recommendation-system
Install miniconda. The sample code requires Python 2.7.


wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
export PATH="/home/$USER/miniconda2/bin:$PATH"
Install the Python packages and TensorFlow. This tutorial will run on any 1.x version of TensorFlow.


cd tensorflow-recommendation-wals
conda create -y -n tfrec
conda install -y -n tfrec --file conda.txt
source activate tfrec
pip install -r requirements.txt
pip install tensorflow==1.15


### Upload sample data to BigQuery

You can use the BigQuery API to upload your data to BigQuery.

### Install the recserve endpoint

This step can take several minutes to complete. You can do this in a separate shell.  That way you can deploy the Airflow service in parallel.  Remember to 'source activate recserve' in any new shell that you open, to activate the recserve envionment.

    source activate recserve

1. Create the App Engine app in your project:

	   gcloud app create --region=us-east1

2. Prepare the deploy template for the Cloud Endpoint API:

	   cd scripts
	   ./prepare_deploy_api.sh                         # Prepare config file for the API.

This will output somthing like:

    ...
    To deploy:  gcloud endpoints services deploy /var/folders/1m/r3slmhp92074pzdhhfjvnw0m00dhhl/T/tmp.n6QVl5hO.yaml

3. Run the endpoints deploy command output above:

	   gcloud endpoints services deploy [TEMP_FILE]

4. Prepare the deploy template for the App Engine App:

	   ./prepare_deploy_app.sh

You can ignore the script output "ERROR: (gcloud.app.create) The project [...] already contains an App Engine application. You can deploy your application using `gcloud app deploy`."  This is expected.

The script will output something like:

	   ...
	   To deploy:  gcloud -q app deploy ../app/app_template.yaml_deploy.yaml

5. Run the command above:

	   gcloud -q app deploy ../app/app_template.yaml_deploy.yaml

This will take several minutes.

	   cd ..

### Deploy the Airflow service

#### Option 1 (recommended): Use Cloud Composer
Cloud Composer is the GCP managed service for Airflow. It is in beta at the time this code is published.

1. Create a new Cloud Composer environment in your project:

    CC_ENV=composer-recserve

    gcloud composer environments create $CC_ENV --location us-central1

This process takes a few minutes to complete.

2. Get the name of the Cloud Storage bucket created for you by Cloud Composer:

    gcloud composer environments describe $CC_ENV \
      --location us-central1 --format="csv[no-heading](config.dagGcsPrefix)" | sed 's/.\{5\}$//'

In the output, you see the location of the Cloud Storage bucket, like this:

    gs://[region-environment_name-random_id-bucket]

This bucket contains subfolders for DAGs and plugins.

3. Set a shell variable that contains the path to that output:

    export AIRFLOW_BUCKET="gs://[region-environment_name-random_id-bucket]"

4. Copy the DAG training.py file to the dags folder in your Cloud Composer bucket:

    gsutil cp airflow/dags/training.py ${AIRFLOW_BUCKET}/dags

5. Import the solution plugins to your composer environment:

    gcloud composer environments storage plugins import \
      --location us-central1 --environment ${CC_ENV} --source airflow/plugins/


#### Option 2: Create an Airflow cluster running on GKE

This can be done in parallel with the app deploy step in a different shell.

1. Deploy the Airflow service using the script in airflow/deploy:

	   source activate recserve
	   cd airflow/deploy
	   ./deploy_airflow.sh

This will take a few minutes to complete.

2. Create "dags," "logs" and "plugins" folders in the GCS bucket created by the deploy script named (managed-airflow-{random hex value}), e.g. gs://managed-airflow-e0c99374808c4d4e8002e481. See https://storage.googleapis.com/solutions-public-assets/recommendation-tensorflow/images/airflow_buckets.png.  The name of the bucket is available in the ID field of the airflow/deploy/deployment-settings.yaml file created by the deploy script.  You can create the folders in the cloud console, or use the following script:

	   cd ../..
	   python airflow/deploy/create_buckets.py

3. Copy training.py to the dags folder in your airflow bucket:

	   export AIRFLOW_BUCKET=`python -c "\
	   import yaml;\
	   f = open('airflow/deploy/deployment-settings.yaml');\
	   settings=yaml.load(f);\
	   f.close();\
	   print settings['id']"`

	   gsutil cp airflow/dags/training.py gs://${AIRFLOW_BUCKET}/dags

4. Copy plugins to the plugins folder of your airflow bucket:

	   gsutil cp -r airflow/plugins gs://${AIRFLOW_BUCKET}

5. Restart the airflow webserver pod

	   WS_POD=`kubectl get pod | grep -o airflow-webserver-[0-9a-z]*-[0-9a-z]*`
	   kubectl get pod ${WS_POD} -o yaml | kubectl replace --force -f -



### Airflow

The Airflow web console can be used to update the schedule for the DAG, inspect logs, manually
execute tasks, etc.

#### Option 1 (Cloud Composer)

Note that after creating the Cloud Composer environment, it takes approximately 25
minutes for the web interface to finish hosting and become accessible.

Type this command to print the URL for the Cloud Composer web console:

    gcloud composer environments describe $CC_ENV --location us-central1 \
        --format="csv[no-heading](config.airflow_uri)"

You see output that looks like the following:

    https://x6c9aa336e72ad0dd-tp.appspot.com

To access the Airflow console for your Cloud Composer instance, go to the URL displayed in the output.

#### Option 2 (GKE)
You can find the URL and login credentials for the airflow admin interface in the file
airflow/deploy/deployment-settings.yaml.

e.g.

    ...
    web_ui_password: IiDYrpwJcT...
    web_ui_url: http://35.226.101.220:8080
    web_ui_username: airflow

The Airflow service can also be accessed from the airflow-webserver pod in the GKE cluster.
Open your project console, navigate to the "Discovery and load balancing" page in GKE, and
click on the endpoint link for the airflow-webserver to access the Airflow admin app.


### Shiny app

I create a simple Shiny app to test. The code is in Shiny directory.