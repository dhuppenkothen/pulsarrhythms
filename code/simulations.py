import numpy as np
import librosa


def gaussian(x, a, loc, w):
    return a*np.exp(-np.log(np.sqrt(2.*np.pi*w**2.))-((x-loc)**2./w**2.))

def simulate_lightcurve(ngaussians, nbins, nperiod=1000):
    period = 2.0
    w = 0.05 * period
    
    x = np.linspace(0, 1, nbins, endpoint=False)
    time, counts = [], []
    
    time = np.zeros(nperiod*len(x))
    counts = np.zeros(nperiod*len(x))
    
    params = [ngaussians]
    for j in range(ngaussians):
        a = np.random.uniform(0.5, 4.0)
        loc = np.random.uniform(0.2, 0.8)
        width = w/np.random.uniform(1.0, 3.0)
        params.extend([a, loc, width])
        
        for i in range(nperiod):
            counts[i*len(x):(i+1)*len(x)] += gaussian(x, a, loc, width)
            time[i*len(x):(i+1)*len(x)] = i*period+period*x

    for i in range(nperiod):
        counts[i*len(x):(i+1)*len(x)] += np.random.normal(0, 0.5,size=len(x))

    counts = np.array(counts)
    time = np.array(time)
    return time, counts, params


def compute_stm(counts, nbins=300, ncycles=5):    
    start_ind = 0
    nsamples = nbins*ncycles
    end_ind = start_ind + nsamples
    n_acf = 1500
    n_stm = 750 #int(nsamples/4.)
    ac_all, scale_all = [], []

    while end_ind <= len(counts):
        ac = librosa.autocorrelate(counts[start_ind:end_ind], max_size=n_acf)
        ac = librosa.util.normalize(ac, norm=np.inf)
        ac_all.append(ac)

        scale = librosa.fmt(ac, n_fmt=n_stm)
        scale_all.append(scale)

        start_ind += nsamples
        end_ind += nsamples

    ac_all = np.array(ac_all)
    scale_all = np.array(scale_all)
    
    return ac_all, scale_all

def simulate(niter=10, fileroot="test"):
    nbins = 300 # number of bins in each cycle
    ncycles = 20
    scale_mean_all, scale_std_all, acf_all, params_all = [], [], [], []

    for i in range(niter):
        print("I am on iteration %i"%i)

        # pick number of Gaussians
        ngaussians = np.random.choice([1,2,3])
        print("Number of Gaussians: " + str(ngaussians))
        # simulate light curve
        time, counts, par = simulate_lightcurve(ngaussians, nbins)

        # add parameters to list
        params_all.append(par[0])

        # compute acf and scale
        ac, scale = compute_stm(counts, ncycles=ncycles)

        # compute mean and std of each scale
        scale_mean = np.mean(np.abs(scale), axis=0)
        scale_std = np.std(np.abs(scale), axis=0)

        # append means and variances to the list
        scale_mean_all.append(scale_mean)
        scale_std_all.append(scale_std)

    # make into numpy arrays
    scale_mean_all = np.array(scale_mean_all)
    scale_std_all = np.array(scale_std_all)

    #try:
        # save results to file
    np.savetxt("../data/%s_stm_mean.txt"%fileroot, scale_mean_all)
    np.savetxt("../data/%s_stm_std.txt"%fileroot, scale_std_all)
    np.savetxt("../data/%s_params.txt"%fileroot, params_all)
    #except:
    #    print("Saving to file failed! Must save by hand!")

    return scale_mean_all, scale_std_all, params_all 
 
def main():
   niter = 50
   fileroot = "test3"
   scale_mean_all, scale_std_all, params_all = simulate(niter, fileroot=fileroot)

   return

if __name__ == "__main__":
   main()
