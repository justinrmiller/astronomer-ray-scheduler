from datetime import datetime
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from google.cloud import aiplatform
from ray.job_submission import JobSubmissionClient
from ray import job_submission
from vertex_ray import Resources

import logging
import ray
import time
import vertex_ray

def cluster_creation(ti):
    try:
        cluster_name = "cluster"
        project_name = "<project-name>"

        logging.info(f"Running cluster creation commands for cluster: {cluster_name}")

        # Define a default CPU cluster, machine_type is n1-standard-16 for the head node, n1-standard-8 for the worker node
        head_node_type = Resources(
            machine_type="n1-standard-16",
            node_count=1
        )

        worker_node_types = [Resources(
            machine_type="n1-standard-8",
            node_count=1,
        )]

        aiplatform.init(project=project_name)

        # Check to see if cluster already exists
        clusters = vertex_ray.list_ray_clusters()
        for cluster in clusters:
            if cluster.cluster_resource_name.split('/')[-1] == cluster_name:
                logging.info(f"Cluster already exists, skipping cluster creation: {cluster.cluster_resource_name}")

                ti.xcom_push(key="cluster_resource_name", value=cluster.cluster_resource_name)

                return "Cluster already exists."

        # Initialize Vertex AI to retrieve projects for downstream operations.
        # Create the Ray cluster on Vertex AI
        cluster_resource_name = vertex_ray.create_ray_cluster(
            head_node_type=head_node_type,
            worker_node_types=worker_node_types,
            python_version="3.10",           # Optional
            ray_version="2.33",              # Optional
            cluster_name=cluster_name,
            enable_metrics_collection=True,  # Optional. Enable metrics collection for monitoring.
        )

        logging.info(f"Cluster successfully created: {cluster_resource_name}")

        ti.xcom_push(key="cluster_resource_name", value=cluster_resource_name)

        return "Cluster creation commands ran successfully."
    except Exception as e:
        logging.error(f"Cluster creation commands failed: {e}")
        raise

def job_submission_script(ti):
    try:
        logging.info("Running job submission script...")

        cluster_resource_name = ti.xcom_pull(key="cluster_resource_name", task_ids="cluster_creation")

        print(f"Submitting job to cluster: {cluster_resource_name}")

        client = JobSubmissionClient("vertex_ray://{}".format(cluster_resource_name))

        job_id = client.submit_job(
            # Entrypoint shell command to execute
            entrypoint="python __main__.py",
            # Path to the local directory that contains the my_script.py file.
            runtime_env={
                "working_dir": "./ray_job",
                "pip": [
                    "numpy",
                    "setuptools<70.0.0",
                    "ray==2.33.0",  # pin the Ray version to the same version as the cluster
                ]
            }
        )

        while True:
            status = client.get_job_status(job_id)
            print(f"Job status: {status}")
            if status in {job_submission.JobStatus.SUCCEEDED, job_submission.JobStatus.FAILED}:
                break
            time.sleep(10)  # Check every 10 seconds

        if status == job_submission.JobStatus.SUCCEEDED:
            print("Job completed successfully.")
        else:
            print("Job failed.")

        return "Job submission script ran successfully."
    except Exception as e:
        logging.error("Job submission failed: %s", e)
        raise

def cluster_deletion(ti):
    try:
        cluster_resource_name = ti.xcom_pull(key="cluster_resource_name", task_ids="cluster_creation")

        logging.info(f"Terminating cluster: {cluster_resource_name}")

        ray.shutdown()
        vertex_ray.delete_ray_cluster(cluster_resource_name)

        return "Cluster deletion commands ran successfully."
    except Exception as e:
        logging.error("Cluster deletion commands failed: %s", e)
        raise

@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    description='A DAG for Ray cluster management and job submission',
    doc_md=__doc__,
    default_args={"owner": "Astro", "retries": 3},
    tags=["ray", "job management"],
)
def ray_vertex_ai_workflow():
    cluster_create = PythonOperator(
        task_id='cluster_creation',
        python_callable=cluster_creation,
    )

    submit_job = PythonOperator(
        task_id='submit_job',
        python_callable=job_submission_script,
    )

    teardown_cluster = PythonOperator(
        task_id='cluster_deletion',
        python_callable=cluster_deletion,
    )

    cluster_create >> submit_job >> teardown_cluster

ray_vertex_ai_workflow()
