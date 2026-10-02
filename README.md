# Lightweight Beam Index Map Using Coupled Gaussian Mixture Models

This repository contains the code for the paper:

[1] A. Kasibovic, F. Weißer, B. Böck, and W. Utschick, “Lightweight Beam Index Map Using Coupled Gaussian Mixture Models,” in _2026 29th International Workshop on Smart Antennas (WSA)_, 2026.

## Running the code
After cloning the repository, the code can be executed by running the `main.py` script manually or by running it in a Docker container.

You can specify custom arguments, such as the used dataset, SNR values, and other. 
To see the full list of arguments run:
```sh
python3 main.py --help
```

### Manual Execution
The `main.py` script was tested with `python 3.12.4` and the libraries specified in the `requirements.txt` file.  

### Docker
The code can be run in a Docker container by executing the following commands.
1. Build the docker image with the necessary requirements:
    ```sh
    docker build -t lightweight-beam-index-mapping .
    ```
2. Run a shell console inside the container:
    ```sh
    docker run --rm -it lightweight-beam-index-mapping /bin/sh
    ```
3. Run the `main.py` script inside the container.
4. Results are stored in the `results` subfolder inside the container.
