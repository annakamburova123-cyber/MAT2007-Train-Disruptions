Seasonal patterns in Dutch train disruptions  
Analysis of whether train disruptions in the Netherlands are worse in some seasons than in others (2022-2025).


Dependencies

Python 3 with pandas, NumPy and matplotlib. Install them with:  
pip install pandas numpy matplotlib

The script uses the following imports:  
from pathlib import Path  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt


Data

Download these four files from Rijden de Treinen open data (https://www.rijdendetreinen.nl/en/open-data/disruptions):  
https://opendata.rijdendetreinen.nl/public/disruptions/disruptions-2022.csv  
https://opendata.rijdendetreinen.nl/public/disruptions/disruptions-2023.csv  
https://opendata.rijdendetreinen.nl/public/disruptions/disruptions-2024.csv  
https://opendata.rijdendetreinen.nl/public/disruptions/disruptions-2025.csv

Put them, with these exact names, in a folder called data next to Project.py.

No manual cleaning is needed; the script removes duplicates and invalid rows itself.


Run

python Project.py

Results are printed in the terminal. The graphs and tables are saved in a new output folder, which the script creates automatically.
