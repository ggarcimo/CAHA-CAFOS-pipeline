# CAHA-CAFOS-Pipeline

**Pipeline for the reduction of spectroscopic images from the CAFOS instrument at the Calar Alto Observatory (CAHA).**

## Overview

This project provides a pipeline to automate the reduction of spectroscopic images obtained using the CAFOS instrument on the 2.2m telescope at the Calar Alto Observatory (CAHA). The pipeline streamlines the process of transforming raw spectroscopic data into calibrated, science-ready data.

Based on UCM pipelines by Jaime Zamorano and Nico Cardiel (UCM).

Adapted by Maria Montguió, Hugo Traning, Nadejda Blagorodnova and Gerard Garcia (UB).

## Features

- **Semiautomated Reduction:** Automatic Bias substraction, flat-field correction, and wavelength calibration. Sky substraction, alignment and spectrum extraction require some human input.
- **Standard flux calibration** is also provided. Files with absolute flux standards provided by CAHA are stored in the Standard Files directory.

## Installation

To set up the pipeline, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/GerardGM99/CAHA-CAFOS-pipeline.git
   cd CAHA-CAFOS-pipeline
   ```

2. **Set Up the Environment:**

   It is recommended to use [Anaconda](https://www.anaconda.com/products/distribution) to manage the environment. Create a new environment and install the required packages:

   ```bash
   conda create --name cafos_pipeline python=3.9
   conda activate cafos_pipeline
   ```

3. **Install Dependencies:**

   Install the necessary Python packages using `pip`:

   ```bash
   pip install -r pip-requirements.txt
   ```

   Alternatively, if you prefer using Conda, you can create the environment using the provided `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   conda activate cafos_pipeline
   ```

## Usage

1. **Prepare Your Data:**

   Put your raw spectroscopic images and associated calibration files (bias, flats, arcs) in a new folder inside the working directory.

2. **Setup:**

   Execute the main script to start the reduction process:

   ```bash
   python
   import CAHA_CAFOS_pipe.py as caf
   ```

   Give as inputs the name of the folder were you stored your data (science and calibration files) and the starting letter or letters of the files you want to reduce.

4. **Run the reduction:**

   ```bash
   caf.general_calibrations() # Bias, flats and wavelength calibrations (automatic)
   caf.science() # Sky substraction, alignment and spectrum extraction (requires human inputs)
   caf.flux_calibration() # Standard flux calibration
   ```
