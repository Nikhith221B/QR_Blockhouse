How to Run
Follow these steps to replicate the analysis and results.

Step 1: Setup
Clone the repository.

Create the data directories: FROG, CRWV, and SOUN.

Place the raw CSV data files into their corresponding ticker directories.

Install dependencies: pip install pandas numpy matplotlib scipy

Step 2: Run Market Impact Analysis

Execute the que1_implementation.py script from your terminal. This will process all the raw data and generate the *_impact_parameters.csv files needed for the next step.

python que1_implementation.py

Step 3: Run the Optimal Execution Algorithm

Once the parameter files have been generated, run the que2_implementation.py script.

python que2_implementation.py

This script will load the parameters, run the dynamic programming optimizer for a sample 100,000 share order, print the final schedule to the console, and display a plot of the aggregated daily execution strategy.

