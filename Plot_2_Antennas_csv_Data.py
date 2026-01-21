# Import necessary packages
from sunpy.net import Fido, attrs as a
from sunpy import timeseries as ts
from sunpy.time import parse_time
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import numpy as np
from astropy import units as u
from astropy.convolution import convolve, Box1DKernel
import matplotlib.dates as mdates
import matplotlib
#%%
#--- USER INPUT ---
#- Select Day of observation
#day = input("Date of observation (format year/month/day): ") #in format year/month/year
#
#day__ = day_.replace("-","_")

#station = input("From which station do you want data amongst the following list?\n-DHO38\n-FTA\n-GBZ\n-NAA\n-NDT\n")

#--- Download VLF data ---
print("Write date in format year/month/day")
day = input("Which day do you wish to study : ")

c= "/" # character
i= day.find(c)
res = day[i+1:]
m = res[0]+res[1]
d = res[3]+res[4]

day_ = day.replace("/", "-")
day__ = day.replace("/", "_")



#%%

#--- Collect GOES X-ray data ---
# The res line gives a list of data from all the satellites that fit the descriptions below
# a.Time("start", "end")  Selects data within the timeframe indicated (here it's entire day)
# a.Instrument("XRS")     Selects data from the wanted instrument on the satellite
# a.Resolution("flx1s")    Selects data with the wanted resolution, here its every second
# a.goes.SatelliteNumber(16) Selects data from wanted satellite
print("\nDownloading GOES data, please wait.")
res0 =  Fido.search(a.Time(day_+" 00:00", day_+" 23:59"), a.Instrument("XRS")) 
print(res0)

resolution ="flx1s"
#satellite_num = int(input("\nPick the GOES satellite number: "))

res = Fido.search(a.Time(day_+" 00:00", day_+" 23:59"), a.Instrument("XRS"), a.Resolution(resolution), a.goes.SatelliteNumber(18)) 
goes_file = Fido.fetch(res) # saves the file to current working directory


#%%
# Print to verify only one dataset corresponds to descriptions above
res
#%%
# Turn the GOES data into a timeseries and have a peek at what it looks like (peek())
goes_ts = ts.TimeSeries(goes_file)

#%%
#-- Define number of antennas used --

num_anten = 2
#int(input("How many antennas do you wish to use? "))



    


def read_vlf_data(file):
    """
    Read VLF csv files and return a pandas Series
    """
    aa = pd.read_csv(file, comment="#", names=["date", "data"])
    sid = pd.Series(20*np.log10(aa["data"].values), index=pd.to_datetime(aa["date"]))
    
    #- This snippet collects the sampling frequency from the comments at the top of the csv
    with open(file, "r") as f:
        for i, line in enumerate(f, start=1):
            if i==11:
                for token in line.split():
                                if token.isdigit():
                                    sf = int(token)
                                    break

    return sid, sf


    
#%%


station_list = ["DHO38", "FTA", "GBZ", "NAA", "NDT"]

#--- Loop through all 5 stations ---
for s, station in enumerate(station_list):
    data_list = [] #will contain data for both antennas
    #-- Collect VLF data for station --
    if num_anten == 1:
        try:
            data = f"/Users/oxytank/Desktop/Capstone/Week_5/{d}_{m}/A1/Dunsink_{station}_{day_}.csv"
            data_list.append(data)
        except:
            data = f"/Users/oxytank/Desktop/Capstone/Week_5/{d}_{m}/Dunsink_{station}_{day_}.csv"
            data_list.append(data)

    elif num_anten == 2:
        data1 = f"/Users/oxytank/Desktop/Capstone/Week_5/{d}_{m}/A1/Dunsink_{station}_{day_}.csv"
        data2 = f"/Users/oxytank/Desktop/Capstone/Week_5/{d}_{m}/A2/Dunsink_{station}_{day_}.csv"
        data_list.append(data1)
        data_list.append(data2)
    
    
    
#%%    
    
    vlf_list = [] #will contain opened VLF data
    sf_list = [] #will contain sampling frequencies

    for dd,data in enumerate(data_list):
        vlf_data, sf = read_vlf_data(data) #read vlf data
        vlf_list.append(vlf_data)
        sf_list.append(sf)


    #--- Plot all GOES and VLF data together ---
    
    # GOES on top and VLF below
    
    #-- Loop through all antennas used ---
    
    
    #- If there is GOES data - 
    #"""
    fig, ax = plt.subplots(2,1, sharex=True)
    goes_ts.plot(axes=ax[0]) #plot GOES
    ax[0].legend( loc='center left', bbox_to_anchor=(1.05, 0.6), title="GOES", fancybox=False, edgecolor="k")
    
    for antenna in range(num_anten):
        colour = "mediumorchid" if antenna==0 else "royalblue"
        ax[1].plot(vlf_list[antenna], color=colour, label=f"SuperSID {antenna+1}")
    ax[1].set_xlim(vlf_list[antenna].index[0], vlf_list[antenna].index[-1])
    ax[1].legend()
    ax[1].set_ylabel("VLF Signal Strength (dB)")
    #ax[1].set_yticks(np.arange(0,1.1,0.2))
        
    ax[1].legend( loc='center left', bbox_to_anchor=(1, 0.55), title=f"{station} VLF Transmitter", fancybox=False, edgecolor="k")
    ax[1].grid()
    plt.savefig(f"{day__}_Plot_{station}.png", bbox_inches="tight")
    plt.show()
    #"""
    
    #- If there is no GOES data -
    """

    plt.figure(figsize=(8,4))
    for antenna in range(num_anten):
        colour = "mediumorchid" if antenna==0 else "royalblue"
        opacity = 1 if antenna==1 else 0.8
        plt.plot(vlf_list[antenna], color=colour, label=f"SuperSID {antenna+1}", alpha=opacity)
    plt.xlim(vlf_list[antenna].index[0], vlf_list[antenna].index[-1])
    plt.legend()
    plt.ylabel("VLF Signal Strength (dB)")
        #plt.yticks(np.arange(0,1.1,0.2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    ax = plt.gca()
    plt.text(
        1.0, -0.1,                # position (x, y) in axes fraction coords
        day,
        transform=ax.transAxes,    # so it's relative to the axes, not data
        ha='right', va='top',
        fontsize=10, color='k'
    )
        
    plt.legend()
    plt.grid()
    plt.title(f"{station} Transmitter")
    plt.savefig(f"{day__}_Plot_{station}.png", bbox_inches="tight")
    plt.show()
    """
    
    #%%
    """
    #- Number of flares
    num_flares =int( input("\nNumber of flares identified: ")) # USER INPUT
    
    #- Collect time of occurence
    
    def flare_func(ind):
        start = input(f"\nAt what time does flare {ind+1} start ? ")
        end = input(f"At what time does flare {ind+1} end ? ")
        t = [start, end]
        return t
    
    if num_flares >0:
        print("\nWrite time in format 08:00")
        flare_list = [flare_func(i) for i in range(num_flares)]
    
    
    #%%
    
    #-- Zoom in on flares --
    
    # First, collect data around each flare peak
    
    # Go through all 3 flares

    try :
        for i in range(num_flares):
            # Save data within relevant time range
            fl_goes = goes_ts.truncate(day +" "+flare_list[i][0],day +" "+ flare_list[i][1])
            fl_obs1 = vlf_data[day +" "+ flare_list[i][0]:  day + " "+ flare_list[i][1]]
    
            
            # Plot
            fig, ax = plt.subplots(1, sharex=True)
            fl_goes.plot(axes=ax[0])
    
    
            ax[0].plot(fl_obs1, color="mediumorchid", label=station, alpha=0.8)
            ax[0].legend(loc="lower right")
            ax[0].grid()
            #ax[0].set_ylim(0,1.1)
            #ax[0].set_ylabel("VLF Normalised Signal Strength")
            fig.suptitle(f"{day} Flare")
            
            #plt.savefig(f"{day__}_Noise_Range.png", bbox_inches="tight")
            plt.savefig(f"{day__}_Flare.png", bbox_inches="tight")
            plt.show()
            
    except:
        print(f"There are no flares on {day}")

    
    
    #%%
    
    # Attempt at normalising GOES data to overplot VLF
    
    #Turn GOES timeseries to dataframe
    if num_flares >0:
        goes_dat = fl_goes.to_dataframe()
        
        norm_goes_A = (goes_dat["xrsa"] - goes_dat["xrsa"].min()) / (goes_dat["xrsa"].max() - goes_dat["xrsa"].min())
        norm_goes_B = (goes_dat["xrsb"] - goes_dat["xrsb"].min()) / (goes_dat["xrsb"].max() - goes_dat["xrsb"].min())
        
        fig, ax1 = plt.subplots()
        ax1.plot(goes_dat["xrsa"], color="b", label="0.5-4.0 $\AA$")
        ax1.plot(goes_dat["xrsb"], color="r", label="1.0-8.0 $\AA$")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.set_ylabel("$W\cdot m^{-2}$")
        ax1.set_yscale("log")
        ax1.set_ylim(10**(-9), 5*10**(-3))
        ax1.legend(loc="upper left")
        
        
        ax2 = ax1.twinx()
        ax2.plot(fl_obs1, color="mediumorchid", label=f"{station} VLF")
        ax2.set_ylabel("VLF Normalised Signal Strength")
        ax2.set_ylim(0,1.1)
        ax2.legend(loc="upper right")
        
        ax = plt.gca()
        plt.text(
            1.0, -0.1,                # position (x, y) in axes fraction coords
            day,
            transform=ax.transAxes,    # so it's relative to the axes, not data
            ha='right', va='top',
            fontsize=10, color='k'
        )
        ax1.grid()
        plt.savefig(f"{day__}_Flare_Overplot.png", bbox_inches="tight")
    
    """
    #%%
"""    
    def noise_func(ind):
        start = input(f"At what time does noise range {ind+1} start ? ") if ind==0 else input(f"\nAt what time does noise range {ind+1} start ? ")
        end = input(f"At what time does noise range {ind+1} end ? ")
        t = [start, end]
        return t
    
    fft_results = []
    ffreq_results = []
    
    #- Function that gets FFT of wanted data
    def fft_func(data, sampling):
        fft_results.append(np.fft.fft(data))
        ffreq_results.append(np.fft.ffreq(len(data), sampling))
        
    
    
    
    #--- Study noise of image ---
    
    #study_noise = input("\nDo you wish to study the noise of this data? ").upper()
    study_noise = "YES"
    if study_noise == "YES":
        
        #-- Loop through antennas --
        fig, ax = plt.subplots(1, sharey=True) #create figure
        
        for antenna in range(num_anten):
            colour = "mediumorchid" if antenna==0 else "royalblue"
        #-- Collect and plot FFT of each antenna --

            noise_dat = vlf_list[antenna]
            freq_list = []
            
                #- 1st truncate data within wanted time range
            noise_obs = noise_dat[day +" 12:00":  day + " 18:00"]  
            fft_result = np.fft.fft(noise_obs)
            ffreq_result = np.fft.fftfreq(len(noise_obs),sf_list[antenna])
                
                #- 2nd Try to smoothen data
            smooth = convolve(noise_obs, Box1DKernel(11))
            fft_smooth = np.fft.fft(smooth)
            ffreq_smooth = np.fft.fftfreq(len(smooth), sf_list[antenna])
                
                #- 3rd plot FFT data of observatory
            opacity = 1 if antenna==1 else 0.8    
            ax.plot(np.abs(ffreq_result), 2*np.abs(fft_result)/len(fft_result),
                       color=colour
                       , label=f"SuperSID {antenna+1}", alpha=opacity)

            ax.set_ylabel("PSD")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(loc="upper right")
            ax.grid()
            ax.set_xlabel("Frequency [Hz]")
            ax.set_xlim(5*10**(-4),10**(-2))
            
            fig.suptitle(f"FFT for {day} 12:00-18:00 {station} ")
            plt.savefig(f"{day__}_FFT_{station}.png", bbox_inches="tight")
    
        #plt.show()
        
        
 """   

#print("\nEnjoy your plots :)")


#%%
"""
#KEEP FOR LATER

    #- Normalise Amplitude
    def norm_func(file):
        # Dropping all the rows with inf values
        file.replace([np.inf, -np.inf], np.nan, inplace=True)
        file.dropna(inplace=True)
        normed =  (file- file.min()) / (file.max() - file.min()) #formula that normalises data
        return normed


    norm_A1 = norm_func(vlf_list[0]) #normalise
    norm_A2 = norm_func(vlf_list[1])
    norm_list = [norm_A1, norm_A2]
#-- Try to combine signals from both antennas to reduce noise --
    A1 = vlf_list[0]
    A2 = vlf_list[1]
    vlf_mix = [float((norm_A1[i] + norm_A2[i])/2) for i in range(len(A1))]
    
    x = [i for i in range(len(A1))]
    plt.figure(figsize=(8,4))
    
    for antenna in range(num_anten):
        colour = "mediumorchid" if antenna==0 else "royalblue"
        opacity = 1 if antenna==1 else 0.8
        plt.plot(x,norm_list[antenna].values, color=colour, label=f"Antenna {antenna+1}", alpha=opacity)
    
    plt.plot(x,vlf_mix, color="r", label="Mixed")
    #plt.xlim(vlf_list[antenna].index[0], vlf_list[antenna].index[-1])
    plt.legend()
    plt.ylabel("VLF Amplitude (dB)")
        #plt.yticks(np.arange(0,1.1,0.2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    ax = plt.gca()
    plt.text(
        1.0, -0.1,                # position (x, y) in axes fraction coords
        day,
        transform=ax.transAxes,    # so it's relative to the axes, not data
        ha='right', va='top',
        fontsize=10, color='k'
    )
        
    plt.legend()
    plt.grid()
    plt.title(f"{station} Transmitter")
    #plt.savefig(f"{day__}_Plot_{station}.png", bbox_inches="tight")
    plt.show()
"""


