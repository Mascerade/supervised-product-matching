import pandas as pd
import numpy as np
import os

""" LOCAL IMPORTS """
from src.data_creation.general_cpu_data_creation import create_general_cpu_data
from src.data_creation.general_drive_data import create_final_drive_data
from src.data_creation.gs_data_creation import create_computer_data
from src.data_creation.laptop_data_creation import create_laptop_data
from src.data_creation.pcpartpicker_data_creation import create_pcpartpicker_data
from src.data_creation.spec_creation import create_spec_laptop_data

# Run the functions
create_general_cpu_data()
create_final_drive_data()
create_computer_data()
create_laptop_data()
create_pcpartpicker_data()
create_spec_laptop_data()
