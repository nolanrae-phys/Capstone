# Capstone

Flare_Class.py:
- contains the code used to collect VLF, GOES and MagIE data within a selected time window. 
- used to obtain all results except for plots of daily data and noise analysis

Plot_2_Antennas_csv_data.py:
- plots daily GOES and VLF data
- VLF data is obtained by opening a locally stored csv file. This can be changed to get data from a url
- can plot specific flare times
- can get FFT of VLF data for wanted time windows
