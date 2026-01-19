# Import necessary packages
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy import timeseries as ts
from sunpy.time import parse_time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import urllib

from astropy import units as u
from astropy.convolution import convolve, Box1DKernel
import matplotlib.dates as mdates
from datetime import timezone
from astropy.time import Time
from datetime import timedelta
import matplotlib.colors as mcolors
#%%y

#-- Objective of code --

# Collect list of flares detected by GOES within wanted time frame
# Then collect VLF data to look at identified flares and try to classify them (A, B, C, M, X)
# Might then try to smoothen data in order to identify X-ray variability post flare peak
#----------------------


#- Pick wanted time frame
print("Time format: year/month/day")
tstart = input("Start of window (inclusive): ")
tend = input("\nEnd of window (exclusive): ")

#%%

#- Get list of flares within window from GOES
event_type = "FL" # say you are looking for flares

result = Fido.search(a.Time(tstart, tend),
                     a.hek.EventType(event_type),
                     a.hek.FL.GOESCls > "C1.0", #threshold magnitude
                     a.hek.OBS.Observatory == "GOES")

hek_results = result["hek"]

# Get flare start, end and peak times along with class
filtered_results = hek_results["event_starttime",
                               "event_endtime", "fl_goescls", "event_peaktime"]
# can work with this like a dataframe

#print(filtered_results)

#%%
# Collect wanted data in variables
fl_start = filtered_results["event_starttime"] #can't access in variable explorer but can work with it (ie print)
fl_end = filtered_results["event_endtime"]
# individual time values are 'astropy.time.core.Time' class

fl_classes = filtered_results["fl_goescls"] # get flare class
fl_peak = filtered_results["event_peaktime"] # get flare peak time


#%%
#should only save flares between 06:00Z and 18:00Z approximately to get daytime VLF data

# collect start, end, peak hours of each event
start_h = np.array([t.datetime.hour for t in fl_start])
end_h = np.array([t.datetime.hour for t in fl_end])


# create masks to only keep data from daytime (06:00-18:00)
start_day_mask = np.array([(h >= 7) & (h < 18) for h in start_h])
end_day_mask = np.array([(h > 7) & (h <= 18) for h in end_h])

daytime = start_day_mask & end_day_mask

day_fl = filtered_results[daytime] #should ouput list of flare times and classes during day
#print("\nDaytime flares identified")
#print(day_fl)

#%%

#- Usual functions to open VLF files and normalise data

def read_vlf_data(file):
    """
    Read VLF csv files and return a pandas Series
    """
    aa = pd.read_csv(file, comment="#", names=["date", "data"])
    sid = pd.Series(20*np.log10(aa["data"].values), index=pd.to_datetime(aa["date"]))
    #
    #- This snippet collects the sampling frequency from the comments at the top of the csv
    with open(file, "r") as f:
        for i, line in enumerate(f, start=1):
            if i==11:
                for token in line.split():
                                if token.isdigit():
                                    sf = int(token)
                                    break

    return sid, sf


#- Normalise signal of file
def norm_func(file):
    # Dropping all the rows with inf values
    file.replace([np.inf, -np.inf], np.nan, inplace=True)
    file.dropna(inplace=True)
    normed =  (file- file.min()) / (file.max() - file.min()) #formula that normalises data

    return normed


def norm_arr(arr):
    """
    Input: array
    Output: normalised array
    """
    res = (arr - arr.min()) / (arr.max() - arr.min())
    return res





def plot_magie(year, month, day):
    
    """
    Plots MAGIE data from https://data.magie.ie along x,y and z axis
    
    INPUT:
        Should be string objects
    
    OUTPUT:
        Figure containing plot of magie data along all 3 axis over entire day     
    """
    
    year = str(year)
    month = f"{int(month):02d}"
    day = f"{int(day):02d}"
    
    base_magie_url = f"https://data.magie.ie/{year}/{month}/{day}/txt/dun{year}{month}{day}.txt"
    
    magie_data = pd.read_csv(base_magie_url, sep="\t", names=["Date", "Index", "Bx", "By", "Bz"], comment=";") #collect txt file from link
    magie_data["Date"] = pd.to_datetime(magie_data["Date"]) #turn Date column content into datetime
    
    time = magie_data["Date"]

    #- Set conditions for Bx, By, Bz -
    colours = ["r", "b", "g"]
    labels = ["x", "y", "z"]
    
    #-- Start Plotting data for xyz axes --
    formatter = matplotlib.dates.DateFormatter('%H:%M') #to format x axis
    
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(6,6))
    ax[0].set_title(f"Dunsink {day}/{month}/{year}")
    for i in range(3):
        ax[i].plot(magie_data["Date"], magie_data[f"B{labels[i]}"], color=colours[i])
        ax[i].set_ylabel(f"$B_{labels[i]}$ (nT)")
        ax[i].grid(linestyle="--")
        ax[i].xaxis.set_major_formatter(formatter)
        plt.setp(ax[i].get_xticklabels(), rotation = 15)
        #ax[i].axvline(flare_time, color="k", linestyle="..", linewidth=1, alpha=0.6)
    ax[0].set_xlim(time.iloc[0], time.iloc[-1])
        
      
    ax = plt.gca()
    plt.text(
        1.025, -0.1,                # position (x, y) in axes fraction coords
        "UTC",
        transform=ax.transAxes,    # so it's relative to the axes, not data
        ha='right', va='top',
        fontsize=10, color='k'
    )
    
   # plt.savefig(f"{day}_{month}_{year}_xyz_MAGIE_Plot.png", bbox_inches="tight")
    
    
    """
    Horizontal component H is what we are interested in.
    Use Pythagoras, horizontal component is sqrt(Bx^2 + By^2)
    This should have the closest link to VLF data
    """
    H = np.sqrt((magie_data["Bx"])**2 + (magie_data["By"])**2)
    
    fig, ax = plt.subplots()
    ax.set_title(f"Dunsink {day}/{month}/{year}")
    ax.plot(magie_data["Date"], H, color="royalblue")
    ax.set_ylabel("$H$ (nT)")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(linestyle="--")
    plt.setp(ax.get_xticklabels(), rotation = 15)
    ax.set_xlim(time.iloc[0], time.iloc[-1])
    plt.text(
        1.025, -0.1,                # position (x, y) in axes fraction coords
        "UTC",
        transform=ax.transAxes,    # so it's relative to the axes, not data
        ha='right', va='top',
        fontsize=10, color='k'
    )
    
    plt.savefig(f"{day}_{month}_{year}_H_MAGIE_Plot.png", bbox_inches="tight")
    

    # Now save MAGIE data
    
    #First add H to MAGIE dataframe                       
    magie_data.insert(5, "H", H, True)
    
    return magie_data
  
    
    

#%%

#--- Loop through each flare ---


"""
Create a dataframe containing information on each flare in window.
- date of flare (year, month, day, start hour, end hour, start minute, end minute)
- flare type (eg. class type: X, class val:5)
- GOES and VLF data
- time delay
- aligned GOES and VLF data
- normalised GOES and VLF data

Overplots VLF on GOES data

"""


prev_day = 0
k = 0

#- Create dataframe to store information about flares for each VLF transmitter and GOES-


df_DHO38 = pd.DataFrame(columns = ["year", "month", "day", "start_hour", "end_hour", "start_minute","end_minute", 
                             "class_type","class_val",
                             "GOES_data", "VLF_data", "time_delay", "GOES_aligned", "VLF_aligned"])

"""
df_FTA = pd.DataFrame(columns = ["year", "month", "day", "start_hour", "end_hour", "start_minute","end_minute", 
                             "class_type","class_val",
                             "GOES_data", "VLF_data", "time_delay", "GOES_aligned", "VLF_aligned"])
"""

df_GBZ = pd.DataFrame(columns = ["year", "month", "day", "start_hour", "end_hour", "start_minute","end_minute", 
                             "class_type","class_val",
                             "GOES_data", "VLF_data", "time_delay", "GOES_aligned", "VLF_aligned"])

df_NAA = pd.DataFrame(columns = ["year", "month", "day", "start_hour", "end_hour", "start_minute","end_minute", 
                             "class_type","class_val",
                             "GOES_data", "VLF_data", "time_delay", "GOES_aligned", "VLF_aligned"])

df_NDT = pd.DataFrame(columns = ["year", "month", "day", "start_hour", "end_hour", "start_minute","end_minute", 
                             "class_type","class_val",
                             "GOES_data", "VLF_data", "time_delay", "GOES_aligned", "VLF_aligned"])



#"time_delay",
#"GOES_aligned","VLF_aligned",
#"GOES_norm", "VLF_norm"

# want time of flare,  class and data for now



# Loop through each flare of the flare list
for flare in day_fl:
    start, end = flare["event_starttime"], flare["event_endtime"] # collect start and end time of flare
    # collect exact date, hour, minute and s of flare
    year = start.datetime.year
    month = start.datetime.month
    day = start.datetime.day
    if len(str(day))<2:
        day = "0"+str(day)
        day_int =int( day)
    else:
        day=str(day)
        day_int = int(day)
        
    peaktime = flare["event_peaktime"]
        
    start_hour, end_hour = start.datetime.hour, end.datetime.hour
    start_minute, end_minute = start.datetime.minute, end.datetime.minute
    start_s, end_s = start.datetime.second, end.datetime.second
    fl_class = flare["fl_goescls"] # collect class of flare
    
    
    # Download data
    if day != prev_day: #only collect data for different days
        k = 0
        # Download GOES
        res = Fido.search(a.Time(f"{year}-{month}-{day} 00:00", f"{year}-{month}-{day} 23:59"),
                          a.Instrument("XRS"), a.Resolution("flx1s"), a.goes.SatelliteNumber(18)) 
        goes_file = Fido.fetch(res)
        goes_ts = ts.TimeSeries(goes_file)
        
        #--- Download VLF from device ---
        # For all 4 transmitters (ignore FTA because data is bad)
        if day_int> 23 or month>=11:
            path_DHO38 = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/A1/Dunsink_DHO38_{year}-{month}-{day}.csv"
            path_GBZ = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/A1/Dunsink_GBZ_{year}-{month}-{day}.csv"
            path_NAA = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/A1/Dunsink_NAA_{year}-{month}-{day}.csv"
            path_NDT = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/A1/Dunsink_NDT_{year}-{month}-{day}.csv"
        else:
            path_DHO38 = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/Dunsink_DHO38_{year}-{month}-{day}.csv"
            path_GBZ =f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/Dunsink_GBZ_{year}-{month}-{day}.csv"
            path_NAA = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/Dunsink_NAA_{year}-{month}-{day}.csv"
            path_NDT = f"/Users/oxytank/Desktop/Capstone/Week_5/{day}_{month}/Dunsink_NDT_{year}-{month}-{day}.csv"
            
        vlf_data_DHO38, sf = read_vlf_data(path_DHO38) #read vlf data
        vlf_data_GBZ, sf = read_vlf_data(path_GBZ) #read vlf data
        vlf_data_NAA, sf = read_vlf_data(path_NAA) #read vlf data
        vlf_data_NDT, sf = read_vlf_data(path_NDT) #read vlf data

        

    else: #if same day as before then use previously downloaded data, makes code much faster
        k+=1

    # start plot 15min before and end 45min after flare times
    t0 = start - 5*u.min
    t1 = end + 15*u.min

    #start_val = vlf_data[start]

    #-- Look at specific flare time --
    fl_goes = goes_ts.to_dataframe() # turn series to dataframes
    fl_vlf_DHO38 = vlf_data_DHO38.to_frame()
    fl_vlf_GBZ = vlf_data_GBZ.to_frame()
    fl_vlf_NAA = vlf_data_NAA.to_frame()
    fl_vlf_NDT = vlf_data_NDT.to_frame()
    
    
    
    start_dt = start.to_datetime(timezone=None) #t0 start
    end_dt = t1.to_datetime(timezone=None) #t1 end
    fl_goes = fl_goes.truncate(before=start_dt, after=end_dt)
    
    fl_vlf_DHO38 = fl_vlf_DHO38.truncate(before=start_dt, after=end_dt)
    fl_vlf_GBZ = fl_vlf_GBZ.truncate(before=start_dt, after=end_dt)
    fl_vlf_NAA = fl_vlf_NAA.truncate(before=start_dt, after=end_dt)
    fl_vlf_NDT = fl_vlf_NDT.truncate(before=start_dt, after=end_dt)
    

    #--- Take absolute value of upside down flares ---
    peak_pdtime = peaktime.to_datetime()
    fl_vlf_series_DHO38 = fl_vlf_DHO38.iloc[:, 0]
    fl_vlf_series_GBZ = fl_vlf_GBZ.iloc[:, 0]
    fl_vlf_series_NAA = fl_vlf_NAA.iloc[:, 0]
    fl_vlf_series_NDT = fl_vlf_NDT.iloc[:, 0]
    
    peak_idx = fl_vlf_series_NAA.index.get_indexer([peak_pdtime], method="nearest")[0] #get index of flare peaktime
    
    series_list = [fl_vlf_series_DHO38, fl_vlf_series_GBZ, fl_vlf_series_NAA, fl_vlf_series_NDT]

    for series in series_list:
        if series.iloc[0] > series.iloc[peak_idx]:
            series *= -1
        
    


    #-- Plot flare --
    """
    fig, ax1 = plt.subplots()
    ax1.plot(fl_goes["xrsa"], color="b", label="0.5-4.0 $\AA$")
    ax1.plot(fl_goes["xrsb"], color="r", label="1.0-8.0 $\AA$")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_ylabel("GOES Flux ($W\cdot m^{-2}$)")
    #ax1.set_yscale("log")
    #ax1.set_ylim(10**(-7), 5*10**(-6))
    ax1.legend( loc='center left', bbox_to_anchor=(1.1, 0.65), title="GOES", fancybox=False, edgecolor="k")


    ax2 = ax1.twinx()
    ax2.plot(fl_vlf_NAA, color="mediumorchid", label="NAA")
    ax2.set_ylabel("VLF Signal Strength (dB)")
    #ax2.set_ylim(0,1.1)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5), title="VLF", fancybox=False, edgecolor="k")

    ax = plt.gca()
    plt.text(
        1.0, -0.1,                # position (x, y) in axes fraction coords
        f"{year}-{month}-{day}",
        transform=ax.transAxes,    # so it's relative to the axes, not data
        ha='right', va='top',
        fontsize=10, color='k'
    )
    ax1.grid()
    plt.suptitle(f"Flare {k+1} ({fl_class} class)")
    #plt.savefig(f"{year}_{month}_{day}_Flare_{k+1}.png", bbox_inches="tight")
    """
    
    # add data to dataframe
    #if np.mean(fl_vlf) >0:
    new_rows_DHO38 = pd.DataFrame({
                            "year":[year],
                             "month":[month],
                             "day":[day],
                             "start_hour":[start.datetime.hour], #t0
                             "end_hour":[end.datetime.hour], #t1
                             "start_minute":[start.datetime.minute], #t0
                             "end_minute":[end.datetime.minute], #t1
                             "class_type":[fl_class[0]],
                             "class_val":[float(fl_class[1:])],
                             "GOES_data":[fl_goes],
                             "VLF_data":[fl_vlf_DHO38],
                             })
                            
    
    df_DHO38= pd.concat([df_DHO38, new_rows_DHO38], ignore_index=True)
    
    new_rows_GBZ = pd.DataFrame({
                            "year":[year],
                             "month":[month],
                             "day":[day],
                             "start_hour":[start.datetime.hour], #t0
                             "end_hour":[end.datetime.hour], #t1
                             "start_minute":[start.datetime.minute], #t0
                             "end_minute":[end.datetime.minute], #t1
                             "class_type":[fl_class[0]],
                             "class_val":[float(fl_class[1:])],
                             "GOES_data":[fl_goes],
                             "VLF_data":[fl_vlf_GBZ],
                             })
                            
    
    df_GBZ= pd.concat([df_GBZ, new_rows_GBZ], ignore_index=True)
    
    
    new_rows_NAA = pd.DataFrame({
                            "year":[year],
                             "month":[month],
                             "day":[day],
                             "start_hour":[start.datetime.hour], #t0
                             "end_hour":[end.datetime.hour], #t1
                             "start_minute":[start.datetime.minute], #t0
                             "end_minute":[end.datetime.minute], #t1
                             "class_type":[fl_class[0]],
                             "class_val":[float(fl_class[1:])],
                             "GOES_data":[fl_goes],
                             "VLF_data":[fl_vlf_NAA],
                             })
                            
    
    df_NAA= pd.concat([df_NAA, new_rows_NAA], ignore_index=True)
    
    
    new_rows_NDT = pd.DataFrame({
                            "year":[year],
                             "month":[month],
                             "day":[day],
                             "start_hour":[start.datetime.hour], #t0
                             "end_hour":[end.datetime.hour], #t1
                             "start_minute":[start.datetime.minute], #t0
                             "end_minute":[end.datetime.minute], #t1
                             "class_type":[fl_class[0]],
                             "class_val":[float(fl_class[1:])],
                             "GOES_data":[fl_goes],
                             "VLF_data":[fl_vlf_NDT],
                             })
                            
    
    df_NDT= pd.concat([df_NDT, new_rows_NDT], ignore_index=True)
    
    
    
    prev_day = day 



#%%

# set this up here for plotting x=y line
x = np.arange(0, 3e-5, (3e-5)/(25))
y = np.arange(0,35,35/25)

#-- Only select M flares or strong C flares--

M_df = df_NAA.query('class_type == "C" or class_type == "M" or class_type == "X" ').reset_index(drop=True)
#or class_val >= 7
    
    

#%%

"""
Aligning Method using Cross-Correlation
"""

#"""
#loop through flares
previous_day = 0

df_list = [df_DHO38, df_GBZ, df_NAA, df_NDT]
transmitter_names = ["DHO38", "GBZ", "NAA", "NDT"]
q=0
for df in df_list:
    # Create new columns
    df["GOES_aligned"] = None
    df["VLF_aligned"] = None
    df["time_delay"] = np.nan
    df["corrcoef"] = np.nan

    for i, row in df.iterrows():
        
        #- Collect current day-
        current_day = row["day"]
        
        if current_day != previous_day:
            count = 1
        elif current_day == previous_day:
            count+=1
        
        
        # GOES sf = 1s
        # VLF sf = 5s
        # -> match GOES and VLF data sizes by taking GOES data for every 5s
        goes_5s = row["GOES_data"].iloc[::5]
        
        #only save values for GOES and VLF, no need for time
        goes_vals = [float(j) for j in goes_5s["xrsb"].values] 
        
        #- Get VLF values for all transmitters
        vlf_vals = np.asarray(row["VLF_data"], dtype=float)#[float(i) for i in row["VLF_data"].values] #only contains signal info for vlf
        vlf_vals = vlf_vals[:-1] # remove last value from VLF for size match
    
        #-- Cross-Correlation to match both signals --
    
        # Turn data into array for easier manipulation with cross correaltion
        vlf_arr = np.asarray(vlf_vals, dtype=np.float64).ravel()
        goes_arr = np.asarray(goes_vals, dtype=np.float64).ravel()
    
    
    
        # Remove median
        vlf_demeaned = vlf_arr - np.median(vlf_arr)
        goes_demeaned = goes_arr - np.median(goes_arr)
    
        # Compute cross-correlation (full mode)
        corr = np.correlate(vlf_demeaned, goes_demeaned, mode='full')
    
    
        # Compute lag array (in samples)
        lags = np.arange(-len(goes_arr) + 1, len(vlf_arr))
    
        # Find the lag that maximizes the correlation
        best_lag = lags[np.argmax(corr)]
        #print(f"Best lag (samples): {best_lag}")
    
        # Convert to seconds (data is sampled every 5 s)
        sampling_period = 5  # seconds
        time_delay = best_lag * 5
        
    
        # Align signals based on best lag
        if best_lag > 0:
            # vlf lags behind GOES (VLF reacts later)
            vlf_aligned = vlf_arr[best_lag:]
            goes_aligned = goes_arr[:len(vlf_aligned)]
        elif best_lag < 0:
            # GOES lags behind VLF (rare)
            goes_aligned = goes_arr[-best_lag:]
            vlf_aligned = vlf_arr[:len(goes_aligned)]
        else:
            vlf_aligned = vlf_arr
            goes_aligned = goes_arr
            
        corrcoef = np.corrcoef(vlf_aligned, goes_aligned)[0,1]
        #print(f"Correlation: {corrcoef}")
    
        #- Make sure aligned data is an array (can cause issues otherwise)
        goes_aligned = np.asarray(goes_aligned, dtype=float).ravel()
        vlf_aligned  = np.asarray(vlf_aligned, dtype=float).ravel()
        
        # make sure aligned data has multiple values
        if len(goes_aligned) < 3 or len(vlf_aligned) < 3:
            df.at[i, "GOES_aligned"] = None
            df.at[i, "VLF_aligned"] = None
        
        else:
        
            #- Save aligned data -
            df.at[i, "GOES_aligned"] = goes_aligned
            df.at[i, "VLF_aligned"] = vlf_aligned
            
            
            #- Save time delay in seconds
            df.at[i, "time_delay"] = time_delay
            
            #- Save correlation coefficient
            df.at[i, "corrcoef"] = corrcoef
        
            # Time axis (adjusted)
            t_aligned = np.arange(len(goes_aligned)) * sampling_period
            ticks = np.arange(0, t_aligned[-1], 900)
        
        
        """
            #- Plot the aligned data -
            if q ==2:
                fig, ax1 = plt.subplots()
                ax1.plot(t_aligned, goes_aligned, color="r", label="1.0–8.0 Å")
                ax1.set_ylabel("GOES Flux ($W\cdot m^{-2}$)")
                ax1.set_xlabel("Time (min)")
                ax1.set_xticks(ticks, labels=(ticks/60).astype(int))
                    
                
                ax2 = ax1.twinx()
                ax2.plot(t_aligned, vlf_aligned, color="mediumorchid", label=transmitter_names[q])
                ax2.set_ylabel("VLF Signal Strength (dB)")
                    
                plt.title(f'Aligned {row["day"]}/{row["month"]}/{row["year"]} {row["class_type"]}{row["class_val"]} Flare ')
                    
                    
                # Legends
                if min(vlf_aligned)>=0:
                    ax1.legend( loc='center left', bbox_to_anchor=(1.1, 0.65), title="GOES", fancybox=False, edgecolor="k")
                    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5), title="VLF", fancybox=False, edgecolor="k")
                else:
                    ax1.legend( loc='center left', bbox_to_anchor=(1.15, 0.65), title="GOES", fancybox=False, edgecolor="k")
                    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 0.5), title="VLF", fancybox=False, edgecolor="k")
                        
                ax = plt.gca()
                plt.text(
                    1.25, 0,                # position (x, y) in axes fraction coords
                    f"Coef: {round(corrcoef,2)}",
                    transform=ax.transAxes,    # so it's relative to the axes, not data
                    ha='right', va='top',
                    fontsize=10, color='k'
                    )
                plt.grid()
                plt.show()
    """
    
        previous_day = current_day
    q += 1
#"""

#%%

"""
Now filter out bad flare matches with a corrcoef <0.7 for example
Also remove flares that do not have data (eg 24/10/2025 )

"""

#Create new dataframes with filters applied
new_df_DHO38 = df_DHO38.query('GOES_aligned != None and VLF_aligned != None and corrcoef >= 0.7 ').reset_index(drop=True)
new_df_GBZ = df_GBZ.query('GOES_aligned != None and VLF_aligned != None and corrcoef >= 0.7 ').reset_index(drop=True)
new_df_NAA = df_NAA.query('GOES_aligned != None and VLF_aligned != None and corrcoef >= 0.7 ').reset_index(drop=True)
new_df_NDT = df_NDT.query('GOES_aligned != None and VLF_aligned != None and corrcoef >= 0.7 ').reset_index(drop=True)

new_df_list = [new_df_DHO38, new_df_GBZ, new_df_NAA, new_df_NDT]


#%%

""" 
Try old method of getting correlation

1. Boxcar smoothing on VLF data
2. Match main peaks of VLF and GOES
3. Collec ∆t between peaks



""" 
previous_day = 0

"""

df_list = [df_DHO38, df_GBZ, df_NAA, df_NDT]
transmitter_names = ["DHO38", "GBZ", "NAA", "NDT"]
z=0
for df in df_list:



    for i, row in df.iterrows():
        #- Collect current day-
        current_day = row["day"]
        
        if current_day != previous_day:
            count = 1
        elif current_day == previous_day:
            count+=1
        
        
        # GOES sf = 1s
        # VLF sf = 5s
        # -> match GOES and VLF data sizes by taking GOES data for every 5s
        goes_5s = row["GOES_data"].iloc[::5]
        
        #only save values for GOES and VLF, no need for time
        goes_vals = [float(j) for j in goes_5s["xrsb"].values] 
        
        #- Get VLF values for all transmitters
        vlf_vals = np.asarray(row["VLF_data"], dtype=float)#[float(i) for i in row["VLF_data"].values] #only contains signal info for vlf
        vlf_vals = vlf_vals[:-1] # remove last value from VLF for size match
        
    
    
        #-- Cross-Correlation to match both signals --
    
        # Turn data into array for easier manipulation
        vlf_arr = np.array(vlf_vals)
    
        #vlf_arr = convolve(vlf_arr, Box1DKernel(50)) #can use boxcar smoothing if necessary
        goes_arr = np.array(goes_vals)
    
    
        # get peak's index
        vlf_pk = np.argmax(vlf_arr)
        
        goes_pk = np.argmax(goes_arr)
        
        dpk = vlf_pk-goes_pk #delta peaks
    
           
        
        if  dpk > 0:
            # vlf lags behind GOES (VLF reacts later)
            vlf_aligned = vlf_arr[dpk:]
            goes_aligned = goes_arr[:len(vlf_aligned)]
            
        elif dpk< 0:
            # GOES lags behind VLF (rare)
            goes_aligned = goes_arr[-dpk:]
            vlf_aligned = vlf_arr[:len(goes_aligned)]
    
        else:
            vlf_aligned = vlf_arr
            goes_aligned = goes_arr
        
        
        #- Check accuracy of match - 
        # Normalise both datasets then take ratio
        norm_goes = norm_arr(goes_aligned)
        norm_vlf = norm_arr(vlf_aligned)
        
        norm_goes = np.asarray(norm_goes).ravel()
        norm_vlf = np.asarray(norm_vlf).ravel()
        
        #print(np.mean(norm_goes))

        
        signal_mask = norm_goes > 0.01   # e.g. 1% of peak

        q_ratio = norm_vlf[signal_mask] / norm_goes[signal_mask]
        print(np.median(q_ratio))
        
        if np.median(q_ratio)<1:
            if np.median(q_ratio)<=0.4:
                match_quality = 1 #1 means good
            else:
                match_quality = 0
                
        elif np.median(q_ratio) >=1:
            if np.median(q_ratio)-1 <=0.4:
                match_quality = 1 
            else:
                match_quality=0
            
        
        
        
        #- Save aligned data -
        df.at[i, "GOES_aligned"] = goes_aligned
    
        
        #- VLF -
        df.at[i, "VLF_aligned"] = vlf_aligned
    
        
        #- Save time delay in seconds -
        df.at[i, "time_delay"] = dpk*5
     
        #- Save match quality
        df.at[i,"match_quality"] = match_quality
        

        #- Plot -
        
        # Time axis (adjusted)
        t_aligned = np.arange(len(goes_aligned)) * 5
        ticks = np.arange(0, t_aligned[-1], 900)
    
        if match_quality ==1 and transmitter_names[z]=="NAA":
        #- Plot raw data -
            fig, ax1 = plt.subplots()
            ax1.plot(np.arange(len(goes_arr)), goes_arr, color="r", label="1.0–8.0 Å")
            ax1.set_ylabel("GOES Flux ($W\cdot m^{-2}$)")
            ax1.set_xlabel("Time (s)")
            #ax1.set_xticks(ticks, labels=(ticks/60).astype(int))
            ax1.legend( loc='center left', bbox_to_anchor=(1.1, 0.65), title="GOES", fancybox=False, edgecolor="k")
        
            ax2 = ax1.twinx()
            ax2.plot(np.arange(len(goes_arr)), vlf_arr, color="mediumorchid", label="VLF")
            ax2.set_ylabel("VLF Signal Strength (dB)")
            ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5), title="VLF", fancybox=False, edgecolor="k")
            plt.title(f'{row["day"]}/{row["month"]}/{row["year"]} {row["class_type"]}{row["class_val"]} Class Flare RAW {transmitter_names[z]}')
            plt.grid()
            plt.show()
        
            #- Plot the aligned data -
            fig, ax1 = plt.subplots()
            ax1.plot(t_aligned, goes_aligned, color="r", label="1.0–8.0 Å")
            ax1.set_ylabel("GOES Flux ($W\cdot m^{-2}$)")
            ax1.set_xlabel("Time (s)")
            #ax1.set_xticks(ticks, labels=(ticks/60).astype(int))
            ax1.legend( loc='center left', bbox_to_anchor=(1.1, 0.65), title="GOES", fancybox=False, edgecolor="k")
        
            ax2 = ax1.twinx()
            ax2.plot(t_aligned, vlf_aligned, color="mediumorchid", label="VLF")
            ax2.set_ylabel("VLF Signal Strength (dB)")
            ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5), title="VLF", fancybox=False, edgecolor="k")
            plt.title(f'{row["day"]}/{row["month"]}/{row["year"]} {row["class_type"]}{row["class_val"]} Class Flare ALIGNED {transmitter_names[z]}')
            plt.grid()
            plt.show()
            
        else:
            continue
        
        previous_day = current_day
        
    z+=1
"""
        
"""
        #- Plot Hysterisis -
        
        vlf_norm = NAA_aligned -NAA_aligned[0]
        goes_norm = goes_aligned -goes_aligned[0]
        plt.figure(figsize=(8,8))
        plt.plot(x,y, color="k", label="y=x $*10^6$", alpha=0.6)
        plt.scatter(goes_norm, vlf_norm, color="b", s=3, label="Cross-correlated", alpha=0.6)
        #plt.scatter(goes_vals, vlf_vals, s=3, color="r", label="Raw", alpha=0.6)
        plt.xlabel("GOES Long X-ray Flux ($W\cdot m^{-2}$)")
        plt.ylabel("VLF Signal Strength (dB)")
        plt.title(f"{row['day']}/{row['month']}/{row['year']} {row['class_type']}{row['class_val']} Class Flare")
        plt.legend(loc="lower right")
        plt.grid()


        if np.mean(vlf_aligned)>0:
            plt.ylim(0.9*min(vlf_norm), 1.1*max(vlf_norm))
        else:
            plt.ylim(0.9*max(vlf_norm), 1.1*min(vlf_norm))
            plt.gca().invert_yaxis()
        plt.xlim(0.9*min(goes_norm),1.1*max(goes_norm))
 

        ax = plt.gca()
        
        plt.text(
            1.0, -0.1,                # position (x, y) in axes fraction coords
            f"{row['day']}/{row['month']}/{row['year']}",
            transform=ax.transAxes,    # so it's relative to the axes, not data
            ha='right', va='top',
            fontsize=10, color='k'
        )
    
        #plt.savefig(f"Short_Hysteresis_{count}_{row["day"]}_{row["month"]}_{row["year"]}.png", bbox_inches="tight")
        plt.show()
 """        
  

#good_matches = (df_NAA["match_quality"] ==1).sum()
#print(f"{good_matches} good matches")

#%%

"""
Try to filter out bad correlations. Some flares are poorly detected by VLF so there is a large difference 
between GOES and VLF curves. Remove these for following section (plotting multiple flares together)

How this should work
1. Collect normalised data
2. Substract VLF from GOES
3. Only keep flares whose substraction is below a threshold
"""

"""
# 1. Collect normalised data
normed_vlf = M_df["VLF_norm"]
normed_goes = M_df["GOES_norm"]

# 2. Substract VLF from GOES
check_corr = []

for i in range(len(normed_vlf)):
    dif = normed_goes[i]-normed_vlf[i] #get difference between VLF and GOES
    
    # 3. Only keep flares within threshold of 0.5
    threshold = 0.032
    print(np.abs(np.median(dif)))
    if np.abs(np.median(dif)) <= threshold:
        check = True
        check_corr.append(check)
    else:
        check = False
        check_corr.append(check)
    
    #Plot data and difference
    plt.figure()
    plt.scatter(np.arange(len(dif)), dif, color="k", alpha=0.6, label="GOES-VLF", s=3)
    plt.plot(np.arange(len(normed_goes[i])), normed_goes[i], color="r", label="GOES")
    plt.plot(np.arange(len(normed_vlf[i])), normed_vlf[i], color="mediumorchid", label="VLF")
    plt.title(f"{M_df['class_type'][i]}{M_df['class_val'][i]}")
    ax=plt.gca()
    plt.text(0.8, 0.05, f"{check} {round(np.abs(np.median(dif)),3)}", transform=ax.transAxes)
    plt.ylim(-1,1.05)
    
M_df["Good_Corr"] = check_corr 
"""

#%%

"""
Compare MAGIE and VLF data

Plots MAGIE vs VLF data (not useful)
"""

"""
prev_day= 0
for i, row in M_df.iterrows():
    
    # Collect and plot MAGIE data
    if row["day"] != prev_day:
        MAGIE_data = plot_magie(row["year"], row["month"], row["day"])
    else:
        continue
    
    
    VLF_dat = row["VLF_data"]
            
    #- Look at MAGIE data for same time duration as VLF flare -
    #1. Collect flare start and end time
    VLF_dat["Date"] = VLF_dat.index
    VLF_dat["Values"] = VLF_dat.iloc[:,0]
    VLF_dat = VLF_dat.reset_index(drop=True)
            
    start_t = VLF_dat["Date"].min()
    end_t = VLF_dat["Date"].max()
            
    prev_day = start_t.day
    
    try:
        mag_dat =  MAGIE_data.set_index("Date") 
        mag_dat = mag_dat.loc[start_t:end_t]
        H = mag_dat["H"].iloc[::5]
                
        plt.figure(figsize=(8,6))
        plt.plot(VLF_dat["Values"], H, alpha=0.6)
        plt.xlabel("VLF Strength (dB)")
        plt.ylabel("$H$ (nT)")
        plt.title(f"{row['day']}/{row['month']}/{row['year']} {row['class_type']}{row['class_val']} Flare")
          
        
    except:
        print("No MAGIE data for this flare")
    
     #in future fix this as consecutive flare days might have the same day of the month but different month (eg 15/10/25 and 15/11/25)
    # this section should only plot full day data once a day!
"""

#%%

"""
Plot all hysteresis together along with VLF=10^6GOES

"""

#-- Try to plot all hysteresis together
w = np.arange(-10,60,1)
g =w*10**-6




number = 0
for df in new_df_list:
    
    plt.figure(figsize=(8,8))
    #plt.plot(g,w, color="k", label="$I_{VLF}=I_{GOES} *10^6$", alpha=0.6)
    max_goes_list = []
    min_goes_list = []
    max_vlf_list = []
    min_vlf_list = []

    for i, row in df.iterrows():
        
        if row["corrcoef"] >= 0.7:
        
            goes = row["GOES_aligned"]
            vlf = row["VLF_aligned"]
    
            
            #max_goes_list.append(max(goes))
            #min_goes_list.append(min(goes))
            #max_vlf_list.append(max(vlf))
            #min_vlf_list.append(min(vlf))
            vlf = [dat -vlf[0] for i,dat in enumerate(vlf)] #remove starting value to "normalise" like Oscar
            goes = [dat -goes[0] for i,dat in enumerate(goes)]
            
            vlf_peak = max(vlf)
            goes_peak = max(goes)
            
            # Give different colour depending on flare class
            if row["class_type"] == "C":
                h_marker="o"
                h_size = 1
                c = "c"
            elif row["class_type"] == "M":
                h_marker="^" 
                h_size=3
                c = "m"
            elif row["class_type"] == "X":
                h_marker="x"
                h_size=6
                c = "r"
            
            if vlf_peak <= 30 and i!=50:
                plt.scatter(goes_peak, vlf_peak, alpha=1, label=f"{row['class_type']}{row['class_val']}", color=c ,marker=h_marker,s=50) 
    
        
    #plt.xlim(0, 1e-2)
    #plt.ylim(-0.11, 60)
    plt.xscale("log")
    plt.xlabel("GOES Long X-ray Flux ($W\cdot m^{-2}$)", fontsize=16)
    plt.ylabel("VLF Signal Strength (dB)", fontsize=16)
    plt.title(f"{tstart} - {tend} {transmitter_names[number]} Transmitter", fontsize=18)
    #plt.ylim(0,20)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='c', linestyle='None',
               markersize=6, label='C class'),
        Line2D([0], [0], marker='^', color='m', linestyle='None',
               markersize=8, label='M class'),
        Line2D([0], [0], marker='x', color='r', linestyle='None',
               markersize=10, label='X class'),
    ]
    
    plt.legend(handles=legend_elements, fancybox=False, fontsize=15, loc="upper left", edgecolor="k", framealpha=0.2)
    
    
    #plt.legend( bbox_to_anchor=(1, 1), title="Flares", fancybox=False, edgecolor="k")
    plt.grid()
    plt.savefig(f"{transmitter_names[number]}_Peaks.png", bbox_inches="tight")
    plt.show()
    
    number +=1



#%%

"""
In this section try to compare aligned data for all 4 transmitters 
and select data from best transmitter for each flare.
Then plot peak VLF vs GOES again but only for the best transmitters selected 

"""

#--- 1. Collect data ---
"""
All transmitters should have the exact same number of flares
so loop through one dataframe to get date of each flare and then

"""


peak_vlf_list = []
peak_goes_list = []
peak_class_list = []
for i, row in df_NAA.iterrows(): #loop through all flares
    
    #- 1. Collect aligned VLF and GOES data from all transmitters for each flare
    try:
        vlf_NAA   = row["VLF_aligned"]
        vlf_DHO38 = df_DHO38.loc[i, "VLF_aligned"]
        vlf_GBZ   = df_GBZ.loc[i, "VLF_aligned"]
        vlf_NDT   = df_NDT.loc[i, "VLF_aligned"]
        
        goes_NAA   = row["GOES_aligned"]
        goes_DHO38 = df_DHO38.loc[i, "GOES_aligned"]
        goes_GBZ   = df_GBZ.loc[i, "GOES_aligned"]
        goes_NDT   = df_NDT.loc[i, "GOES_aligned"]
        
        vlf_list = [vlf_DHO38, vlf_GBZ, vlf_NAA, vlf_NDT]
        goes_list = [goes_DHO38, goes_GBZ, goes_NAA, goes_NDT]
        
        #- 2. Substract VLF from GOES for each transmitter
        res_DHO38 = np.median(norm_arr(vlf_DHO38) - norm_arr(goes_DHO38))
        res_GBZ = np.median(norm_arr(vlf_GBZ) - norm_arr(goes_GBZ))
        res_NAA = np.median(norm_arr(vlf_NAA) - norm_arr(goes_NAA))
        res_NDT = np.median(norm_arr(vlf_NDT) - norm_arr(goes_NDT))
        
        res_list = [np.abs(res_DHO38), np.abs(res_GBZ), np.abs(res_NAA), np.abs(res_NDT)]
        best_match_ind = np.argmin(res_list)
        print(best_match_ind)
        
        """
        if row["class_type"] == "X" and row["class_val"] == 4.0:
            best_match_ind = 3
        
        if row["class_type"] == "X" and row["class_val"] == 5.1:
            best_match_ind = 0
        
        if row["class_type"] == "X" and row["class_val"] == 1.2:
            best_match_ind = 1
        """
        #- 3. Get peak data for best transmitter
        # Get VLF and GOES data from best transmitter
        best_vlf = vlf_list[best_match_ind]
        best_goes = goes_list[best_match_ind]
        # Get peak of VLF and GOES data
        peak_vlf = np.max(best_vlf) - np.min(best_vlf)
        peak_goes = np.max(best_goes) - np.min(best_goes) #don't know if just getting max is good (what if there's a bgger peak before or after flare?)
        
        #- 4. Save data in list for plotting
        
        peak_vlf_list.append(peak_vlf)
        peak_goes_list.append(peak_goes)
        peak_class_list.append(row["class_type"])
        
        # In case plot data for visual inspection
        t = np.arange(len(best_vlf))*5
    
        
        #- Plot the aligned data -
        if row["class_type"] == "X":
            fig, ax1 = plt.subplots()
            ax1.plot(t, best_goes, color="r", label="1.0–8.0 Å")
            ax1.set_ylabel("GOES Flux ($W\cdot m^{-2}$)")
            ax1.set_xlabel("Time (s)")
            #ax1.set_xticks(ticks, labels=(ticks/60).astype(int))
            ax1.legend( loc='center left', bbox_to_anchor=(1.1, 0.65), title="GOES", fancybox=False, edgecolor="k")
        
            ax2 = ax1.twinx()
            ax2.plot(t, best_vlf, color="mediumorchid", label="VLF")
            ax2.set_ylabel("VLF Signal Strength (dB)")
            ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5), title="VLF", fancybox=False, edgecolor="k")
            plt.title(f'{row["day"]}/{row["month"]}/{row["year"]} {row["class_type"]}{row["class_val"]} Flare {transmitter_names[best_match_ind]} Transmitter')
            plt.grid()
            plt.show()
        
    except:
        continue
    



plt.figure(figsize=(8,8))
for i in range(len(peak_vlf_list)):
    if peak_class_list[i] == "C":
        markershape = "o"
        markercolour = "c"
    elif peak_class_list[i] == "M":
        markershape = "^"
        markercolour= "mediumorchid"
    elif peak_class_list[i] == "X":
        markershape="x"
        markercolour="r"
        
    plt.scatter(peak_goes_list[i], peak_vlf_list[i], color=markercolour, marker=markershape, s=50)
plt.xscale("log")
plt.grid()
plt.xlabel("GOES Long X-ray Flux ($W\cdot m^{-2}$)", fontsize=16)
plt.ylabel("VLF Signal Strength (dB)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"{tstart} - {tend} Best Transmitters", fontsize=18)
legend_elements = [
    Line2D([0],[0], color="k", linestyle="-",
           markersize=6, label="$I_{VLF}=I_{GOES} *10^6$"),
    Line2D([0], [0], marker='o', color='c', linestyle='None',
           markersize=6, label='C class'),
    Line2D([0], [0], marker='^', color='m', linestyle='None',
           markersize=8, label='M class'),
    Line2D([0], [0], marker='x', color='r', linestyle='None',
           markersize=10, label='X class'),
]

plt.legend(handles=legend_elements, fancybox=False, fontsize=15, loc="upper left", edgecolor="k", framealpha=0.2)
#plt.xlim(7e-7, 1e-3)


#%%

"""
This section gathers data to create a VLF flare table.
Characteristics we are interesteed in:
    -time delay
    - rise time (of GOES vs VLF maybe)
    
Could plot histograms showing distribution of time delay for each flare type
Maybe look at time of day for flare?
Could work on end of flare here    

"""

#loop through 4 transmitters later
#for now only look at NAA

#- 1. Get time delays for each flare class
z=0
for df in new_df_list:

    
    try:
        C_flares = df[(df["class_type"] == "C") & (df["time_delay"] < 2400 ) ] #ignore flare if peak is at the end of aligned data
        if z!=2:
            M_flares = df[df["class_type"] == "M"] #& (df["match_quality"]==1) 
            X_flares = df[(df["class_type"] == "X") ]
        else:
            M_flares = df[(df["class_type"] == "M") & (df["time_delay"] <2000)]
            X_flares = df[(df["class_type"] == "X") & (df["class_val"] !=4) & (df["time_delay"]<2000)]
        #
        flare_df_list = [C_flares, M_flares, X_flares]
        
        time_delays_df = pd.concat([C_flares["time_delay"], M_flares["time_delay"], X_flares["time_delay"]])
            
        
        """
        plt.figure()
        plt.hist(C_flares["time_delay"], bins=30, color="c", edgecolor="k", alpha=0.8)
        plt.axvline(np.mean(C_flares["time_delay"]), color="k", linestyle="--", 
                    label=f"$\mu$ = {round(np.mean(C_flares['time_delay']),2)} ± {round(np.std(C_flares['time_delay']),2)} s")
        plt.xlabel("Time Delay (s)")
        plt.ylabel("Counts")
        plt.legend()
        
        
        plt.figure()
        plt.hist(M_flares["time_delay"], bins=30, color="mediumorchid", edgecolor="k", alpha=0.8)
        plt.axvline(np.mean(M_flares["time_delay"]), color="k", linestyle="--",
                    label=f"$\mu$ = {round(np.mean(M_flares['time_delay']),2)} ± {round(np.std(M_flares['time_delay']),2)} s")
        plt.xlabel("Time Delay (s)")
        plt.ylabel("Counts")
        plt.legend()
        
        plt.figure()
        plt.hist(X_flares["time_delay"], bins=30, color="r", edgecolor="k", alpha=0.8)
        plt.axvline(np.mean(X_flares["time_delay"]), color="k", linestyle="--",
                    label=f"$\mu$ = {round(np.mean(X_flares['time_delay']),2)} ± {round(np.std(X_flares['time_delay']),2)} s")
        plt.xlabel("Time Delay (s)")
        plt.ylabel("Counts")
        plt.legend()
        """
        
        
        #- 1.2 Recreate Laura's plot
        
        mean_td = np.mean(time_delays_df) #get mean time delay
        std_td = np.std(time_delays_df)
        
        plt.figure(figsize=(8,6))
        
        """
        for i,flare in df_NAA.iterrows():
            
            # collect peak goes value of flare
            goes = flare["GOES_aligned"]
            goes = [dat -goes[0] for j,dat in enumerate(goes)] #start method
            goes_peak = max(goes)
            
            
            if flare["class_type"] == "C":
                flare_colour = "c"
                flare_shape = "o"
                           
            
            elif flare["class_type"] == "M":
                flare_colour = "mediumorchid"
                flare_shape = "^"
                
            elif flare["class_type"] == "X":
                flare_colour = "r"
                flare_shape = "x"
            
            
            plt.scatter(goes_peak, flare["time_delay"], color= flare_colour, marker = flare_shape, s=50)
        """
        
        for i,flare in C_flares.iterrows():
            goes = flare["GOES_aligned"]
            goes = [dat -goes[0] for j,dat in enumerate(goes)] #start method
            goes_peak = max(goes)
            
            plt.scatter(goes_peak, flare["time_delay"], color="c", marker="o", s=50)
        
        
        for i,flare in M_flares.iterrows():
            goes = flare["GOES_aligned"]
            goes = [dat -goes[0] for j,dat in enumerate(goes)] #start method
            goes_peak = max(goes)
            
            plt.scatter(goes_peak, flare["time_delay"], color="mediumorchid", marker="^", s=50)
            
        for i,flare in X_flares.iterrows():
            goes = flare["GOES_aligned"]
            goes = [dat -goes[0] for j,dat in enumerate(goes)] #start method
            goes_peak = max(goes)
            
            plt.scatter(goes_peak, flare["time_delay"], color="r", marker="x", s=50)
            
        
        plt.axhline(mean_td, color="k", linestyle="--", label=f"$\mu$={round(mean_td,2)}±{round(std_td,1)}s", alpha=0.6)
        plt.grid()
        plt.xscale("log")
        plt.xlabel("1-8 $\AA$ X-ray peak flux ($W\cdot m^{-2}$)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.legend(loc="lower right", fontsize=13)
        
        plt.ylabel("Time Delay (min)", fontsize=16)
        tick_minutes = np.arange(-60, 61, 10)     # 0, 10, 20, ..., 100 min
        tick_seconds = tick_minutes * 60 
        plt.yticks(tick_seconds, tick_minutes, fontsize=14)
        #plt.ylim(-900,900)
        plt.title(f"{transmitter_names[z]} transmitter")
    
    except:
        continue
    z+=1


#%%
#- 2. Get rise time of flare
#Define the rise time as the time between the first data point and the flare's peak (ie max value)
z=0
for df in new_df_list:

    for i,flare in df.iterrows():
        try:
            # collect peak goes value of flare
            goes = flare["GOES_aligned"]
            goes = [dat -goes[0] for j,dat in enumerate(goes)] #start method
            goes_peak = max(goes)
            goes_peak_ind = np.argmax(goes)
            #say start of flare is min value before the peak
            #say start time is when signal reaches 20% of peak
            #thresh_goes = 0.2 * goes_peak
            pre_peak_goes = np.array(goes[:goes_peak_ind]) #look at data before peak
            #thresh_goes_mask = pre_peak_goes >= thresh_goes
            #rise_goes = pre_peak_goes[thresh_goes_mask]
            goes_min_ind = np.argmin(pre_peak_goes)
            goes_rise_time =  (abs(goes_peak_ind - goes_min_ind))*5
            #len(rise_goes)*5
            
        
            # collect peak vlf value of flare
            vlf = flare["VLF_aligned"]
            vlf = [dat -vlf[0] for j,dat in enumerate(vlf)] #start method
            vlf_peak = max(vlf)
            vlf_peak_ind = np.argmax(vlf)
            
            #peak of vlf shouldn't be too far from goes once aligned

            #thresh_vlf = 0.2*vlf_peak
            pre_peak_vlf = np.array(vlf[:vlf_peak_ind])
            #thresh_vlf_mask = pre_peak_vlf >= thresh_vlf
            #rise_vlf = pre_peak_vlf[thresh_vlf_mask]
            vlf_min_ind = np.argmin(pre_peak_vlf)
            vlf_rise_time = (abs(vlf_peak_ind-vlf_min_ind))*5
            #len(rise_vlf)*5
            
            
            
            #- Save peak data - 
            df.at[i,"GOES_peak_val"] = goes_peak
            df.at[i,"GOES_peak_ind"] = goes_peak_ind
            df.at[i,"VLF_peak_val"] = vlf_peak
            df.at[i,"VLF_peak_ind"] = vlf_peak_ind
            
            
            #- Save rise time -
            df.at[i,"GOES_rise_time"] = goes_rise_time
            df.at[i,"VLF_rise_time"] = vlf_rise_time
        
        except:
            continue
    
    plt.figure(figsize=(8,6))
    plt.scatter(
        df[df["class_type"] == "C"]["GOES_rise_time"],
        df[df["class_type"] == "C"]["VLF_rise_time"],
        color = "c", marker="o", s=50 
        )
    plt.scatter(
        df[df["class_type"] == "M"]["GOES_rise_time"],
        df[df["class_type"] == "M"]["VLF_rise_time"],
        color = "mediumorchid", marker="^", s=50 
        )
    plt.scatter(
        df[df["class_type"] == "X"]["GOES_rise_time"],
        df[df["class_type"] == "X"]["VLF_rise_time"],
        color = "r", marker="x", s=50 
        )
    
    plt.grid()
    #plt.xscale("log")
    plt.xlabel("GOES 1-8 $\AA$ Rise Time (min)", fontsize=16)
    plt.ylabel("VLF Rise Time (min)", fontsize=16)
    
    tick_minutes = np.arange(-60, 61, 10)     # 0, 10, 20, ..., 100 min
    tick_seconds = tick_minutes * 60 
    plt.yticks(tick_seconds, tick_minutes, fontsize=14)
    plt.xticks(tick_seconds, tick_minutes,fontsize=14)
    plt.ylim(-60, 3000)
    plt.xlim(-60, 3000)
    legend_elements = [
        Line2D([0], [0], marker='o', color='c', linestyle='None',
               markersize=6, label='C class'),
        Line2D([0], [0], marker='^', color='m', linestyle='None',
               markersize=8, label='M class'),
        Line2D([0], [0], marker='x', color='r', linestyle='None',
               markersize=10, label='X class'),
    ]
    plt.title(f"{transmitter_names[z]}")
    plt.legend(handles=legend_elements, fancybox=False, fontsize=15, loc="upper left", edgecolor="k", framealpha=0.2)
    z+=1
    print(
          f"C flare mean rise time for GOES : {round(np.mean(df[df['class_type'] == 'C']['GOES_rise_time']),2)} ± {round(np.std(df[df['class_type'] == 'C']['GOES_rise_time']),2)}s")
    print(
          f"C flare mean rise time for VLF : {round(np.mean(df[df['class_type'] == 'C']['VLF_rise_time']),2)} ± {round(np.std(df[df['class_type'] == 'C']['VLF_rise_time']),2)}s")
    
    print(
          f"M flare mean rise time for GOES : {round(np.mean(df[df['class_type'] == 'M']['GOES_rise_time']),2)} ± {round(np.std(df[df['class_type'] == 'M']['GOES_rise_time']),2)}s")
    print(
          f"M flare mean rise time for VLF : {round(np.mean(df[df['class_type'] == 'M']['VLF_rise_time']),2)} ± {round(np.std(df[df['class_type'] == 'M']['VLF_rise_time']),2)}s")
    
    print(
          f"X flare mean rise time for GOES : {round(np.mean(df[df['class_type'] == 'X']['GOES_rise_time']),2)} ± {round(np.std(df[df['class_type'] == 'X']['GOES_rise_time']),2)}s")
    
    print(
          f"X flare mean rise time for VLF : {round(np.mean(df[df['class_type'] == 'X']['VLF_rise_time']),2)} ± {round(np.std(df[df['class_type'] == 'X']['VLF_rise_time']),2)}s")


#%%
#get GOES data for full day
res = Fido.search(a.Time("2025-11-11", "2025-11-11"), a.Instrument("XRS"), a.Resolution("flx1s"), a.goes.SatelliteNumber(18)) 
goes_file = Fido.fetch(res) # saves the file to current working directory
goes_ts = ts.TimeSeries(goes_file)

path = "/Users/oxytank/Desktop/Capstone/Week_5/11_11/A1/Dunsink_NAA_2025-11-11.csv"

vlf_X5, sf_x5 = read_vlf_data(path)
vlf_X5 = vlf_X5.to_frame()

#%%

"""
This section is for MagIE analysis.

1. Plot GOES and MAGIE data together
2. Get MAGIE delay and rise time for X5 flare

"""


"""
1. Collect MAGIE data for X5 flare duration
"""

X5_flare = new_df_NAA[(new_df_NAA["class_type"] == "X") & (new_df_NAA["class_val"] >= 4 )] #get row of info for X5 flare

year = 2025#str(year)
month = 11# f"{int(month):02d}"
day = 14#f"{int(day):02d}"

base_magie_url = f"https://data.magie.ie/{year}/{month}/{day}/txt/dun{year}{month}{day}.txt"

magie_data = pd.read_csv(base_magie_url,sep="\t",names=["Date", "Index", "Bx", "By", "Bz"],comment=";") #collect txt file from link
magie_data["Date"] = pd.to_datetime(magie_data["Date"]) #turn Date column content into datetime
magie_data = magie_data.set_index("Date").sort_index()



start_dt = pd.Timestamp(
    year=int(X5_flare["year"].item()),
    month=int(X5_flare["month"].item()),
    day=int(X5_flare["day"].item()),
    hour=int(X5_flare["start_hour"].item()),
    minute=int(X5_flare["start_minute"].item())
)

end_dt = pd.Timestamp(
    year=int(X5_flare["year"].item()),
    month=int(X5_flare["month"].item()),
    day=int(X5_flare["day"].item()),
    hour= 11,#int(X5_flare["end_hour"].item()),
    minute=0#int(X5_flare["end_minute"].item())+45
)

magie_X5 = magie_data.truncate(before=start_dt, after=end_dt) #saves MAGIE data for duration of X5 flare
goes_dat = goes_ts.to_dataframe()
goes_X5 = goes_dat["xrsb"]
goes_X5 = goes_X5.truncate(before=start_dt, after=end_dt)
vlf_X5 = vlf_X5.truncate(before=start_dt, after=end_dt)
time = magie_X5.index

 #- Set conditions for Bx, By, Bz -
colours = ["r", "b", "g"]
labels = ["x", "y", "z"]
 
 #-- Start Plotting data for xyz axes --
formatter = matplotlib.dates.DateFormatter('%H:%M') #to format x axis
 
fig, ax = plt.subplots(3,1, sharex=True, figsize=(6,6))
ax[0].set_title(f"Dunsink {day}/{month}/{year}")
for i in range(3):
   ax[i].plot(magie_X5.index, magie_X5[f"B{labels[i]}"], color=colours[i])
   ax[i].set_ylabel(f"$B_{labels[i]}$ (nT)")
   ax[i].grid(linestyle="--")
   ax[i].xaxis.set_major_formatter(formatter)
   plt.setp(ax[i].get_xticklabels(), rotation = 15)
     #ax[i].axvline(flare_time, color="k", linestyle="..", linewidth=1, alpha=0.6)
ax[0].set_xlim(magie_X5.index[0], magie_X5.index[-1])
     
   
ax = plt.gca()
plt.text(
     1.04, -0.25,                # position (x, y) in axes fraction coords
     "UTC",
     transform=ax.transAxes,    # so it's relative to the axes, not data
     ha='right', va='top',
     fontsize=10, color='k'
 )


"""
2. Compare MAGIE and GOES for X5 flare
    will need to apply boxcar smoothing to MAGIE to have a sampling frequency of 5s like GOES and VLF

"""
 
 #vlf_arr = convolve(vlf_arr, Box1DKernel(10)) #can use boxcar smoothing if necessary
goes_x5 = X5_flare["GOES_data"].iloc[0]
goes_x5 = goes_x5["xrsb"]
Bx = convolve(magie_X5["Bx"], Box1DKernel(5), boundary="extend")#the boundary part prevents boundary issues with boxcar smoothing
By = convolve(magie_X5["By"], Box1DKernel(5), boundary="extend")
Bz = convolve(magie_X5["Bz"], Box1DKernel(5), boundary="extend") 
g_norm = norm_arr(goes_x5)
gx = np.arange(len(g_norm))
Bx_norm = norm_arr(Bx)



fig, ax1 = plt.subplots()
ax1.plot(magie_X5.index, Bx, color="r", label="$B_x$ (nT)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.set_ylabel("$B_x$ (nT)")
ax1.legend( loc='center left', bbox_to_anchor=(0, 1.1), title="MagIE", fancybox=False, edgecolor="k")


ax2 = ax1.twinx()
ax2.plot(goes_X5, color="k", label="1.0-8.0 $\AA$", linestyle="--")
ax2.set_ylabel("GOES Flux ($W \cdot m^{-2}$)")
ax2.tick_params(axis='y')
ax2.set_yscale("log")
ax2.legend(loc="upper right", bbox_to_anchor=(1, 1.2), title="GOES", fancybox=False, edgecolor="k")
ax = plt.gca()
plt.text(
    1.0, -0.1,                # position (x, y) in axes fraction coords
    f"{year}-{month}-{day}",
    transform=ax.transAxes,    # so it's relative to the axes, not data
    ha='right', va='top',
    fontsize=10, color='k'
)
ax1.grid(linestyle="--")




#%%

edge1 = pd.Timestamp(
    year=int(X5_flare["year"].item()),
    month=int(X5_flare["month"].item()),
    day=int(X5_flare["day"].item()),
    hour=int(X5_flare["end_hour"].item()),
    minute=int(0),
    second=int(0)
)

edge2 = pd.Timestamp(
    year=int(X5_flare["year"].item()),
    month=int(X5_flare["month"].item()),
    day=int(X5_flare["day"].item()),
    hour=int(X5_flare["end_hour"].item()),
    minute=int(1),
    second=int(48)
)



labels = ["x", "y", "z"]
colours = ["r", "b", "g"]

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
axes[0].set_title(f"Dunsink {day}/{month}/{year}")

magie_handles = []
goes_handle = None

for i, ax1 in enumerate(axes):

    # ---- MagIE axis ----
    h_magie, = ax1.plot(
        magie_X5.index,
        magie_X5[f"B{labels[i]}"],
        color=colours[i],
        label=f"$B_{labels[i]}$ (nT)"
    )
    ax1.set_ylabel(f"$B_{labels[i]}$ (nT)")
    ax1.grid(linestyle=":")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #ax1.axvline(edge1, color="k", linestyle="-", linewidth=1, alpha=0.6)
    #ax1.axvline(edge2, color="k", linestyle="-", linewidth=1, alpha=0.6)
    

    magie_handles.append(h_magie)

    # ---- GOES axis ----
    ax2 = ax1.twinx()
    h_goes, = ax2.plot(
        goes_X5,
        color="k",
        linestyle="--",
        label="GOES 1–8 Å"
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("GOES Flux ($W\cdot m^{-2}$)")

    goes_handle = h_goes  # same for all panels


axes[0].set_xlim(magie_X5.index[0], magie_X5.index[-1])
axes[-1].text(
    1.04, -0.15,
    "UTC",
    transform=axes[-1].transAxes,
    ha="right", va="top",
    fontsize=10
)

#- Place all legends at the bottom of the plot
fig.legend(
    magie_handles,
    [r"$B_x$", r"$B_y$", r"$B_z$"],
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.35, 0.05),
    fancybox=False,
    edgecolor="k"
)

fig.legend(
    [goes_handle],
    ["GOES 1–8 Å"],
    loc="lower center",
    bbox_to_anchor=(0.7, 0.05),
    fancybox=False,
    edgecolor="k"
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

#%%

"""
Get the time delay and rise time by fitting MagIE data
"""

#-1 truncate the data and save it as an array

Bx_rise = magie_X5["Bx"].truncate(before=edge1, after=edge2)
By_rise = magie_X5["By"].truncate(before=edge1, after=edge2)
Bz_rise = magie_X5["Bz"].truncate(before=edge1, after=edge2)

goes_rise =goes_x5.truncate(before=edge1, after=edge2)


delta_t = np.arange(len(Bx_rise))


#-2 Apply linear fit
Bxcoefficients = np.polyfit(delta_t, Bx_rise, 1)
Bycoefficients = np.polyfit(delta_t, By_rise, 1)
Bzcoefficients = np.polyfit(delta_t, Bz_rise, 1)


# Create polynomial function
px = np.poly1d(Bxcoefficients)
py = np.poly1d(Bycoefficients)
pz = np.poly1d(Bzcoefficients)



plt.figure()
plt.scatter(delta_t, Bx_rise, color="r", s=5)
plt.plot(delta_t, px(delta_t), color="k")

print(f"Bx coefficients : {Bxcoefficients}")

plt.figure()
plt.scatter(delta_t, By_rise, color="b", s=5)
plt.plot(delta_t, py(delta_t), color="k")

print(f"By coefficients : {Bycoefficients}")

plt.figure()
plt.scatter(delta_t, Bz_rise, color="g", s=5)
plt.plot(delta_t, pz(delta_t), color="k")

print(f"Bz coefficients : {Bzcoefficients}")


x_err = np.mean(abs(px(delta_t) - Bx_rise))
y_err = np.mean(abs(py(delta_t) - By_rise))
print(f"∆Bx = {round(max(px(delta_t)) - min(px(delta_t)),1)} ± {round(x_err,1)} nT")
print(f"∆By = {round(max(py(delta_t)) - min(py(delta_t)),1)} ± {round(y_err,1)} nT")


#%%

#Plot GOES, VLF and MagIE in a figure


vlf_x5 = X5_flare["VLF_data"].iloc[0]
vlf_X5 = vlf_X5 -vlf_X5.iloc[0]

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 7))


ax[0].plot(goes_X5, color="r", label="GOES 1-8 $\AA$")
ax[0].set_yscale("log")
ax[0].set_ylabel("GOES Flux ($W\cdot m^{-2}$)")
ax[0].grid(linestyle="--")
ax[0].legend(fancybox=False, edgecolor="k", loc="upper right")

ax[1].plot(vlf_X5, color="mediumorchid", label="NAA")
ax[1].set_ylabel("VLF Signal Strength (dB)")
ax[1].grid(linestyle="--")
ax[1].legend(fancybox=False, edgecolor="k", loc="upper right")

ax[2].plot(magie_X5.index,magie_X5["By"], color="b", label="MagIE")
ax[2].grid(linestyle="--")
ax[2].set_ylabel("$B_y$ (nT)")
ax[2].legend(fancybox=False, edgecolor="k", loc="upper right")

ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[2].set_xlim(magie_X5.index[0], magie_X5.index[-1])


ax = plt.gca()
plt.text(
    1.022, -0.15,                # position (x, y) in axes fraction coords
    "UTC",
    transform=ax.transAxes,    # so it's relative to the axes, not data
    ha='right', va='top',
    fontsize=10, color='k'
)

#%%

#plot H for MagIE
H = np.sqrt((magie_X5["Bx"])**2 + (magie_X5["By"])**2)

fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(magie_X5.index, H, color="royalblue")
ax.set_ylabel("H (nT)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlim(magie_X5.index[0], magie_X5.index[-1])
ax.grid(linestyle="--")
ax = plt.gca()
plt.text(
    1.022, -0.1,                # position (x, y) in axes fraction coords
    "UTC",
    transform=ax.transAxes,    # so it's relative to the axes, not data
    ha='right', va='top',
    fontsize=10, color='k'
)



#%%

#count the amount of flares in each df after corrcoef filter
z = 0
for df in new_df_list:
    c_count = len(df[df["class_type"] == "C"])
    m_count = len(df[df["class_type" ]== "M"])
    x_count = len(df[df["class_type"] == "X"])
    
    print(f"{transmitter_names[z]} Transmitter:")
    print(f"C flares : {c_count}")
    print(f"M flares : {m_count}")
    print(f"X flares : {x_count}")
    print("")
    z+=1


