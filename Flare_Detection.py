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
#%%
#--- USER INPUT ---
#- Select Day of observation
day = input("Date of observation (format year/month/day): ") #in format year/month/year
day_ = day.replace("/", "-")
day__ = day_.replace("-","_")
obs_name1 = input("Do you want data from Dunsink or Birr? ") #Pick observatory to be studied
station = input("From which station do you want data amongst the following list?\n-DHO38\n-FTA\n-GBZ\n-NAA\n-NDT\n")

#--- Download VLF data ---
# Base url and file name that will be modified to fit the wanted observatory and date
base_url = "https://vlf.ap.dias.ie/data/observatory/super_sid/date/csv/"
base_file = "Observatory_station_date.csv"

vlf_base_url = base_url.replace("observatory", obs_name1.lower()) 
vlf_base_url = vlf_base_url.replace("date", day)
vlf_file = base_file.replace("Observatory", obs_name1)
vlf_file = vlf_file.replace("station", station)
vlf_file = vlf_file.replace("date", day_)

try:
    urllib.request.urlretrieve(vlf_base_url + vlf_file, vlf_file) #retrieve data using link to web
except:
    print(f"{obs_name1} has no data for that day")

# Name 2nd observatory based on the name of the 1st observatory
if obs_name1 == "Dunsink":
    obs_name2 = "Birr"
elif obs_name1 == "Birr":
    obs_name2 = "Dunsink"


#- Compare with data from observatory 2
obs2 = input(f"\nDo you wish to compare data with {obs_name2} observatory?: ")
if obs2.upper() == "YES":
    try:
        obs2_vlf_base_url = vlf_base_url.replace(obs_name1.lower(), obs_name2.lower()) #change name of observatory in url
        obs2_vlf_file = vlf_file.replace(obs_name1, obs_name2) #change name of observatory in filename
        urllib.request.urlretrieve(obs2_vlf_base_url + obs2_vlf_file, obs2_vlf_file) #retrieve data using link to web
        compare = True
        num_obs = 2
        print(f"\nData found for {obs_name2} observatory ")
    except:
        print(f"\n{obs_name2} observatory does not have data for that day\n")
        compare = False
        num_obs = 1
else:
    compare=False
    num_obs=1


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
satellite_num = int(input("\nPick the GOES satellite number: "))

res = Fido.search(a.Time(day_+" 00:00", day_+" 23:59"), a.Instrument("XRS"), a.Resolution(resolution), a.goes.SatelliteNumber(satellite_num)) 
goes_file = Fido.fetch(res) # saves the file to current working directory

#%%
# Print to verify only one dataset corresponds to descriptions above
res
#%%
# Turn the GOES data into a timeseries and have a peek at what it looks like (peek())
goes_ts = ts.TimeSeries(goes_file)

#%%
#--- Plot VLF data ---

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


#- Normalise Amplitude
def norm_func(file):
    # Dropping all the rows with inf values
    file.replace([np.inf, -np.inf], np.nan, inplace=True)
    file.dropna(inplace=True)
    normed =  (file- file.min()) / (file.max() - file.min()) #formula that normalises data
    return normed


obs1_vlf_data, sf1 = read_vlf_data(vlf_file) #read vlf data
norm_obs1 = norm_func(obs1_vlf_data) #normalise
obs_data = [norm_obs1] #list contains data for all observatories
sf_list = [sf1]

if compare == True:
    obs2_vlf_data, sf2 = read_vlf_data(obs2_vlf_file)
    norm_obs2 = norm_func(obs2_vlf_data)
    obs_data.append(norm_obs2)
    sf_list.append(sf2)

    
#%%

#--- Plot all GOES and VLF data together ---

# GOES on top and VLF below
fig, ax = plt.subplots(2, sharex=True)

goes_ts.plot(axes=ax[0]) #plot GOES

#ax[1].plot(norm_Birr, color='royalblue', label="Birr")
ax[1].plot(norm_obs1, color='mediumorchid' if obs_name1=="Dunsink" else "royalblue", label=obs_name1)
if compare == True:
    ax[1].plot(norm_obs2, color="royalblue" if obs_name2 =="Birr" else "mediumorchid", label=obs_name2, alpha=0.8)
ax[1].set_xlim(obs1_vlf_data.index[0], obs1_vlf_data.index[-1])
ax[1].legend()
#ax[1].set_xlim(Birr_vlf_data.index[0], Birr_vlf_data.index[-1])
ax[1].set_ylabel("VLF Amplitude (db)")
ax[1].set_yticks(np.arange(0,1.1,0.2))
ax[1].legend()
ax[1].grid()
#plt.savefig(f"{day__}_Normed_Plot.png", bbox_inches="tight")
plt.show()

#%%

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
        #fl_birr = norm_Birr[Flares[0][i] :  Flares[1][i]]
        fl_obs1 = norm_obs1[day +" "+ flare_list[i][0]:  day + " "+ flare_list[i][1]]
        if compare == True:
            fl_obs2 = norm_obs2[day +" "+ flare_list[i][0]:  day + " "+ flare_list[i][1]]
        
        # Plot
        fig, ax = plt.subplots(2, sharex=True)
        fl_goes.plot(axes=ax[0])

        if compare == True:
            ax[1].plot(fl_obs2, color="royalblue", label=obs_name2)
        ax[1].plot(fl_obs1, color="mediumorchid", label=obs_name1, alpha=0.8)
        ax[1].legend(loc="upper right")
        fig.suptitle(f"Flare {i+1}")
        #plt.savefig(f"{day__}_Noise_Range.png", bbox_inches="tight")
        #plt.savefig(f"Flare_{i+1}_Comparison.png", bbox_inches="tight")
        plt.show()
        
except:
    print(f"There are no flares on {day}")


#%%
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

study_noise = input("\nDo you wish to study the noise of this data? ").upper()

if study_noise == "YES":
    
    #- Collect noise ranges -
    num_noise_range = int(input("\nHow many noise ranges do you wish to study? "))
    print("\nWrite time in format 08:00")
    noise_list = [noise_func(i) for i in range(num_noise_range)]
    
    colours=[ "mediumorchid", "royalblue"]
    names = [obs_name1, obs_name2]
#%%
    #-- Collect and plot FFT of each observatory for each noise range --
    for k in range(num_noise_range): # loop through amount of noise ranges
        fig, ax = plt.subplots(1,num_obs, sharey=True) #create figure
        if num_obs == 1:
            ax = [ax]
        for j in range(num_obs): #loop through amount of observatories
        
            #- 1st truncate data within wanted time range
            noise_obs = obs_data[j][day +" "+ noise_list[k][0]:  day + " "+ noise_list[k][1]]  
            fft_result = np.fft.fft(noise_obs)
            ffreq_result = np.fft.fftfreq(len(noise_obs),sf_list[j])
            
            #- 2nd Try to smoothen data
            smooth = convolve(noise_obs, Box1DKernel(11))
            fft_smooth = np.fft.fft(smooth)
            ffreq_smooth = np.fft.fftfreq(len(smooth), sf_list[j])
            
            #- 3rd plot FFT data of observatory
            
            ax[j].plot(np.abs(ffreq_result), 2*np.abs(fft_result)/len(fft_result),
                       color="mediumorchid" if names[j] =="Dunsink" else "royalblue"
                       , label=names[j])
            ax[j].plot(np.abs(ffreq_smooth), 2*np.abs(fft_smooth)/len(fft_smooth), color="r", label="Smooth", alpha=0.3) #smooth signal
            
            
            
           # ax[j].axvline(2.73*10**(-3), color="k", alpha=0.3, linestyle="--")
           # ax[j].text(0.42, 0.15, "2.73mHz", transform=ax[j].transAxes)
            
            #ax[j].axvline(4.07*10**(-3), color="k", alpha=0.3, linestyle="--")
            #ax[j].text(0.7, 0.15, "4.07mHz", transform=ax[j].transAxes)
            ax[j].set_ylabel("PSD")
            ax[j].set_xscale("log")
            ax[j].set_yscale("log")
            ax[j].legend(loc="upper right")
            ax[j].grid()
            ax[j].set_xlabel("Frequency [Hz]")
            ax[j].set_xlim(5*10**(-4),10**(-2))
            
            peak_s = [6*10**(-4), 10**(-3)]
            peak_e = [9*10**(-4), 1.5*10**(-3)]
            datx = np.abs(ffreq_result)
            daty = np.abs(fft_result)
            x = [0.01, 0.27, 0.71]
            for i in range(2):
                s = peak_s[i]
                e = peak_e[i]
                mask = (datx <= e) & (datx >=s)
                x_dat= datx[mask]
                y_dat = daty[mask]
                freq = x_dat[np.argmax(y_dat)]
                print(freq)
                ax[j].text(x[i], 0.15,f"{round( freq*10**3,2)} mHz", transform=ax[j].transAxes)
                ax[j].axvline(freq, color="k", alpha=0.3, linestyle="--")

        fig.suptitle(f"FFT for {day} {noise_list[k][0]}-{noise_list[k][1]} Noise Range")
        #plt.savefig(f"FFT_{day__}.png", bbox_inches="tight")

    #plt.show()
    
    
    

print("\nEnjoy your plots :)")





