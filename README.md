# A Flow-based latent state generative model of neural population responses to natural images
Code for "A Flow-based latent state generative model of neural population responses to natural images" submitted to 35th Conference on Neural Information Processing Systems (NeurIPS 2021)

Here we provide the code used to train and evaluate models presented in the paper, along with two example notebooks in the `notebooks` directory to faciliate code usage:
- `example1_neural_data.ipynb`: this notebook includes an example figure of how the learned transformation trained on neural responses compare to other fixed transformations (i.e. sqrt and anscombe), similar to Fig 2b. It also includes code run the Flow-based as well as the control models (Poisson and ZIG) upon dataset availability.
- `example2_synthetic_data.ipynb`: this notebook shows how the synthetic data was generated and includes a demo of training a flow-based model on the synthetic data.

## Requirements
- `docker` and `docker-compose`.

## Intructions to run the code

1. Unzip the supplementary material zip file and place the code folder, named `code`, in your preferred directory
2. Using the shell of your choice, navigate to the code directory and run the following command inside the code directory

    ```bash
    docker-compose run -d -p 10101:8888 notebook
    ```
    This command will create a docker image followed by a docker container from that image in which we can run the code. 

3. You can now open the [jupyter lab evironment](https://jupyterlab.readthedocs.io/en/stable/#) in your browser via `localhost:10101`