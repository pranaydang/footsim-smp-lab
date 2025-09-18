import numpy as np
from scipy import signal
import random

from .classes import Afferent,AfferentPopulation,Stimulus
from .surface import Surface, foot_surface, null_surface
from .constants import affparams

default_params ={'dist':1.,
                 'affclass':Afferent.affclasses,
                 'idx':None,
                 'len':1.,
                 'loc':np.array([0, 0]),
                 'fs':5000.,
                 'ramp_len':0.05,
                 'ramp_type':'lin',
                 'pin_radius':0.5,
                 'pre_indent':0.,
                 'pad_len':0.,
                 'pins_per_mm':10}


def affpop_single_models(**args):

    """Returns AfferentPopulation containing all single neuron models.

    Kwargs:
        affclass: Single affclass or list, e.g. ['SA1','RA'] (default: all).
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    if type(affclass) is not list:
        affclass = [affclass]
    a = AfferentPopulation()
    for t in affclass:
        for i in range(Afferent.affparams.get(t).shape[0]):
            a.afferents.append(Afferent(t,idx=i,**args))
    return a


def affpop_linear(**args):
    """Generates afferents on a line extending from the origin along the chosen axis.

    Kwargs:
        min_dist (float): distance of closest afferent in mm (default 0.).
        max_dist (float): distance of farthest afferent in mm (default: 10.).
        num (int): number of afferents to be placed on line (default: 11)
        affclass (str or list): Single affclass or list (default: ['SA1','RA','PC']).
        idx (int): Afferent model index; None (default) picks all available.
        axis (int): Axis along which afferents will be place (0 or 1, default: 0)
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    if type(affclass) is not list:
        affclass = [affclass]
    dist = args.pop('dist',default_params['dist'])
    min_dist = args.pop('min_dist',0.)
    max_dist = args.pop('max_dist',10.)
    num = args.pop('num',11)
    idx = args.pop('idx',default_params['idx'])
    axis = args.pop('axis',0)

    locs = np.linspace(min_dist,max_dist,num)

    a = AfferentPopulation(**args)

    for l in np.nditer(locs):
        if idx is None:
            if axis==0:
                a_sub = affpop_single_models(location=np.array([l,0]),**args)
            else:
                a_sub = affpop_single_models(location=np.array([0,l]),**args)
            a.afferents.extend(a_sub.afferents)
        else:
            for t in affclass:
                if axis==0:
                    a.afferents.append(
                        Afferent(t,location=np.array([l,0]),idx=idx,**args))
                else:
                    a.afferents.append(
                        Afferent(t,location=np.array([0,l]),idx=idx,**args))
    return a


def affpop_linear_regional(**args):

    """Generates afferents on a line extending from the regional center of a region along the chosen axis.

        Kwargs:

            dist (float): distance between neighboring afferent locations in mm
                (default 1.).
            max_extent (float): distance of farthest afferent in mm (default: 10.).
            affclass (str or list): Single affclass or list (default: ['SA1','RA','PC']).
            idx (int): Afferent model index; None (default) picks all available.
            axis (int): Axis along which afferents will be place (0 or 1, default: 0)
            region = regional tag where the population will be added
            args: All other kwargs will be passed on to Afferent constructor.

        Returns:
            AfferentPopulation object.
        """

    affclass = args.pop('affclass', default_params['affclass'])

    if type(affclass) is not list:

        affclass = [affclass]

    dist = args.pop('dist', default_params['dist'])
    max_extent = args.pop('max_extent', default_params['max_extent'])
    idx = args.pop('idx', default_params['idx'])
    axis = args.pop('axis', 1)
    region = args.pop('region')
    center = foot_surface.centers[foot_surface.tag2idx(str(region))][0]

    locs = np.r_[0.:max_extent + dist:dist]

    a = AfferentPopulation()

    for l in np.nditer(locs):

        if idx is None:

            if axis == 0:
                a_sub = affpop_single_models(location=np.array([center[0]+l, center[1]]), affclass=affclass, **args)

                a_sub = affpop_single_models(location=np.array([center[0]-l, center[1]]), affclass=affclass,
                                                **args)
            else:
                a_sub = affpop_single_models(location=np.array([center[0], center[1]+l]), affclass=affclass, **args)
                a.afferents.extend(a_sub.afferents)

                a_sub = affpop_single_models(location=np.array([center[0], center[1]-l]), affclass=affclass, **args)
                a.afferents.extend(a_sub.afferents)

        else:

            for t in affclass:

                if axis == 0:
                    a.afferents.append(Afferent(t, location=np.array([center[0]+l, center[1]]), idx=idx, **args))
                    a.afferents.append(Afferent(t, location=np.array([center[0]-l, center[1]]), idx=idx, **args))
                else:
                    a.afferents.append(Afferent(t, location=np.array([center[0], center[1]+l]), idx=idx, **args))
                    a.afferents.append(Afferent(t, location=np.array([center[0], center[1]-l]), idx=idx, **args))

    return a


def allregions_affpop_single_models(affclass, idx):

    """ Pins a single model on every surface of the foot sole.

        Arguments:

            affclass (str): afferent model class
            idx (float): afferent model id

        Returns:

           The afferent class, the afferent model and a dictionary with the afferent populations with locations
           as keys
        """

    afferent_populations = dict()

    for i in range(foot_surface.num):

        afferent_populations[foot_surface.locate(foot_surface.centers[i])[0][0]] = AfferentPopulation()

    for i in range(foot_surface.num):

        # Pins the afferents in the right locations and fills the regions dictionary #

        afferent_populations[foot_surface.locate(foot_surface.centers[i])[0][0]].\
            afferents.append(Afferent(affclass, idx=int(idx), location=foot_surface.centers[i]))

    return affclass, idx, afferent_populations


def allregions_affpop_allmodels():

    """ Creates one afferent population of ALL single models per region

     Returns:

        A dictionary with the afferent populations with locations as keys
     """

    afferent_populations = dict()

    for i in range(foot_surface.num):
        afferent_populations[foot_surface.locate(foot_surface.centers[i])[0][0]] = AfferentPopulation()

    for affclass in Afferent.affclasses:

        for i in range(foot_surface.num):

            for idx in range(affparams[affclass].shape[0]):
                # Pins the afferents in the right locations and fills the regions dictionary #

                afferent_populations[foot_surface.locate(foot_surface.centers[i])[0][0]]\
                    .afferents.append(Afferent(affclass, idx=idx, location=foot_surface.centers[i]))

    return afferent_populations


def affpop_grid(**args):
    """Generates afferents on a 2D square grid centred on the origin.

    Kwargs:
        dist (float): distance between neighboring afferent locations in mm
            (default 1.).
        max_extent (float): length of square in mm (default: 10.).
        affclass (str or list): Single affclass or list (default: ['SA1','RA','PC']).
        idx (int): Afferent model index; None (default) picks all available.
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    if type(affclass) is not list:
        affclass = [affclass]
    dist = args.pop('dist',default_params['dist'])
    max_extent = args.pop('max_extent',default_params['max_extent'])
    idx = args.pop('idx',default_params['idx'])

    locs = np.r_[-max_extent/2:max_extent/2+dist:dist]

    a = AfferentPopulation()

    for l1 in np.nditer(locs):
        for l2 in np.nditer(locs):
            if idx is None:
                a_sub = affpop_single_models(
                    location=np.array([l1,l2]),**args)
                a.afferents.extend(a_sub.afferents)
            else:
                for t in affclass:
                    a.afferents.append(
                        Afferent(t,location=np.array([l1,l2]),idx=idx,**args))
    return a


def affpop_foot(**args):
    """Places receptors on the standard foot surface.

    Kwargs:
        affclass (str or list): Single affclass or list, (default: ['SA1','RA','PC']).
        region (str): identifies region(s) to populate, with None selecting
            all regions (default: None), e.g.
            'D2' for a full finger (digit 2), or
            'D2d' for a part (tip of digit 2).
        density (dictionary): Mapping from tags to densities
            (default: foot_surface.density).
        density_multiplier (float): Allows proportional scaling of densities
            (default: 1.).
        high_density (bool): Sets density to newer, higher estimates of innervation
            as published in Corniani and Saal, 2020 (default: False).
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    return affpop_surface(surface=foot_surface,**args)

def affpop_surface(**args):
    """Places receptors on a surface. Like affpop_foot(), but for arbitrary
    surfaces using the surface keyword.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    surface = args.pop('surface',foot_surface)
    density = args.pop('density',surface.density)
    density_multiplier = args.pop('density_multiplier',1.)
    high_density = args.pop('high_density',False)
    if high_density:
        density_multiplier *= 2.35
    if type(affclass) is not list:
        affclass = [affclass]
    region = args.pop('region',None)
    seed = args.pop('seed',None)

    if seed is not None:
        random.seed(seed)

    idx = surface.tag2idx(region)

    afferents = list()
    for a in affclass:
        for i in idx:
            dens = density_multiplier*density[(a,i)]
            xy = surface.sample_uniform(i,density=dens,seed=seed)
            for l in range(xy.shape[0]):
                afferents.append(Afferent(a,location=xy[l,:],**args))
    affpop = AfferentPopulation(surface=surface,*afferents)
    return affpop


def stim_sine(**args):
    """Generates indenting complex sine stimulus.

    Kwargs:
        freq (float or list): vector of frequencies in Hz (default: 200.).
        amp (float or list): vector of amplitudes (zero-to-peak) in mm (default: 0.02).
        phase (float or list): vector of phases in degrees (default: 0.).
        len (float): stimulus duration in s (default: 1.).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default 5000.).
        ramp_len (float): length of on and off ramps in s (default: 0.05).
        ramp_type (str): 'lin' or 'sin' (default: 'lin').
        pin_radius (float): radius of probe pin in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        pad_len (float): duration of stimulus zero-padding (default: 0.).
        args: All other kwargs will be passed on to Stimulus constructor.

    Returns:
        Stimulus object.
    """
    freq = np.array(args.pop('freq',200.))
    amp = np.array(args.pop('amp',.02*np.ones(freq.shape)))
    phase = np.array(args.pop('phase',np.zeros(freq.shape)))
    len = args.pop('len',default_params['len'])
    loc = np.array(args.pop('loc',default_params['loc']))
    fs = args.get('fs',default_params['fs'])
    ramp_len = args.pop('ramp_len',default_params['ramp_len'])
    ramp_type = args.pop('ramp_type',default_params['ramp_type'])
    pre_indent = args.pop('pre_indent',default_params['pre_indent'])
    pad_len = args.pop('pad_len',default_params['pad_len'])

    trace = np.zeros(int(fs*len))
    for f,a,p in zip(np.nditer(freq),np.nditer(amp),np.nditer(phase)):
        trace += a*np.sin(p*np.pi/180. \
            + np.linspace(0.,2.*np.pi*f*len,int(fs*len)))

    apply_ramp(trace,ramp_len=ramp_len,fs=fs)
    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,**args)


def stim_noise(**args):
    """Generates bandpass Gaussian white noise stimulus.

    Kwargs:
        freq (list): upper and lower bandpass frequencies in Hz (default: [100.,300.]).
        amp (float): amplitude (standard deviation of trace) in mm (default: 0.02).
        len (float): stimulus duration in s (default: 1.).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default 5000.).
        ramp_len (float): length of on and off ramps in s (default: 0.05).
        ramp_type (str): 'lin' or 'sin' (default: 'lin').
        pin_radius (float): radius of probe pin in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        pad_len (float): duration of stimulus zero-padding (default: 0.).
        seed (int): seed for random number generator (default: None).
        args: All other kwargs will be passed on to Stimulus constructor.

    Returns:
        Stimulus object.
    """
    freq = args.pop('freq',[100.,300.])
    amp = args.pop('amp',.02)
    len = args.pop('len',default_params['len'])
    loc = np.array(args.pop('loc',default_params['loc']))
    fs = args.pop('fs',default_params['fs'])
    ramp_len = args.pop('ramp_len',default_params['ramp_len'])
    ramp_type = args.pop('ramp_type',default_params['ramp_type'])
    pre_indent = args.pop('pre_indent',default_params['pre_indent'])
    pad_len = args.pop('pad_len',default_params['pad_len'])
    seed = args.pop('seed',None)

    if seed is not None:
        np.random.seed(seed)

    trace = np.random.randn(int(fs*len))

    bfilt,afilt = signal.butter(3,np.array(freq)/fs/2.,btype='bandpass')
    trace = signal.lfilter(bfilt,afilt,trace)

    trace = trace/np.std(trace)*amp

    apply_ramp(trace,ramp_len=ramp_len,fs=fs)
    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,**args)

def stim_linear(len_prev, **args): # written by Pranay Dang
    amp = args.pop('amp',.03)
    len = args.pop('len',0.01)
    loc = np.array(args.pop('loc',default_params['loc']))
    fs = args.pop('fs',default_params['fs'])
    pre_indent = args.pop('pre_indent',default_params['pre_indent'])
    pad_len = args.pop('pad_len',default_params['pad_len'])

    time_step = 0.01 # in seconds
    n = time_step*fs

    trace = np.linspace(len_prev, len_prev+amp, int(n))

    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,**args)


def stim_impulse(**args):
    
    """Generates a short impulse to the skin.

    Kwargs:
        amp (float): amplitude of the pulse in mm (default: 0.03).
        len (float): pulse duration in s (default: 0.01).
        pad_len (float): duration of stimulus zero-padding (default: 0.045).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default 5000.).
        pin_radius (float): radius of probe pin in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        args: All other kwargs will be passed on to Stimulus constructor.

    Returns:
        Stimulus object.
    """
    amp = args.pop('amp',.03)
    len = args.pop('len',0.01)
    loc = np.array(args.pop('loc',default_params['loc']))
    fs = args.pop('fs',default_params['fs'])
    pre_indent = args.pop('pre_indent',default_params['pre_indent'])
    pad_len = args.pop('pad_len',default_params['pad_len'])

    trace = signal.gaussian(int(fs*len),std=7) *\
        np.sin(np.linspace(-np.pi,np.pi,int(fs*len)))
    trace = trace/np.max(trace)*amp

    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,**args)


def stim_ramp(**args):
    """Generates ramp up / hold / ramp down indentation.

    Kwargs:
        amp (float): amplitude in mm (default: 1.).
        ramp_type (str): 'lin' or 'sin' (default: 'lin').
        len (float): total duration of stimulus in s (default: 1.).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default: 5000.).
        ramp_len (float): duration of on and off ramps in s (default 0.05).
        pin_radius (float): probe radius in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        pad_len (float): duration of stimulus zero-padding (default: 0.).
        args: All other kwargs will be passed on to Stimulus constructor.

    Returns:
        Stimulus object.
    """
    amp = args.pop('amp',1.)
    len = args.pop('len',default_params['len'])
    loc = np.array(args.pop('loc',default_params['loc']))
    fs = args.pop('fs',default_params['fs'])
    ramp_len = args.pop('ramp_len',default_params['ramp_len'])
    ramp_type = args.pop('ramp_type',default_params['ramp_type'])
    pre_indent = args.pop('pre_indent',default_params['pre_indent'])
    pad_len = args.pop('pad_len',default_params['pad_len'])

    trace = amp*np.ones(int(fs*len))
    apply_ramp(trace,ramp_len=ramp_len,fs=fs,ramp_type=ramp_type)
    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,**args)


def stim_indent_shape(shape,trace,**args):
    """Applies indentation trace to several pins that make up a shape.

    Args:
        shape (2D array): pin positions making up object shape, e.g. shape_bar().
        trace (array or Stimulus):

    Kwargs:
        rectify (bool): Resets negative indentations to zero (default: True)
        pin_radius (float): probe radius in mm.
        fs (float): sampling frequency.
        args: All other kwargs will be passed on to Stimulus constructor.

    Returns:
        Stimulus object.
     """
    if type(trace) is Stimulus:
        t = trace.trace[0:1]
        if 'fs' not in args:
            args['fs'] = trace.fs
        if 'pin_radius' not in args:
            args['pin_radius'] = trace.pin_radius
    else:
        t = np.reshape(np.atleast_2d(trace),(1,-1))

    t = np.tile(t,(shape.shape[0],1))

    if shape.shape[1]==3:
        t += shape[:,2:3]

    if args.pop('rectify',True):
        t[t<0] = 0

    return Stimulus(trace=t,location=shape[:,0:2],**args)


def shape_bar(**args):
    """Generates pin locations for a bar shape.

    Kwargs:
        width (float): bar width in mm (default: 1.).
        height (float): bar height in mm (default: 0.5).
        angle: bar angle in degrees (default: 0.).
        pins_per_mm (int): Pins per mm (default: 10).
        center (array): Location of stimulus center (default: [0.,0.]).
        hdiff (float): depth difference between center and edge (default: 0.).

    Returns:
        3D array of pin locations.
    """
    width = args.get('width',1.)
    height = args.get('height',.5)
    angle = np.deg2rad(args.get('angle',0.))
    pins_per_mm = args.get('pins_per_mm',default_params['pins_per_mm'])

    xy = np.mgrid[-width/2.:width/2.:width*pins_per_mm*1j,
        -height/2.:height/2.:height*pins_per_mm*1j]
    xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
    xy = np.dot(np.array([[np.cos(angle),-np.sin(angle)],
        [np.sin(angle),np.cos(angle)]]),xy.T).T
    if 'hdiff' in args:
        d =  -args.get('hdiff')*(1/width*2*xy[:,0:1])**2
    else:
        d = np.zeros((xy.shape[0],1))
    d -= np.max(d)
    if 'center' in args:
        xy = xy + np.array(args.get('center'))
    xy = np.hstack((xy,d))
    return xy


def shape_circle(**args):
    """Generates pin locations for a circle.

    Kwargs:
        radius (float): circle radius in mm (default: 2.).
        pins_per_mm (int): Pins per mm (default: 10).
        curvature (float): Between 0 (flat) and 1 (sphere) (default: 0.).
        center (array): Location of stimulus center (default: [0.,0.]).

    Returns:
        3D array of pin locations.
    """

    radius = args.get('radius',2.)
    pins_per_mm = args.get('pins_per_mm',default_params['pins_per_mm'])

    xy = np.mgrid[-radius:radius:2*radius*pins_per_mm*1j,
        -radius:radius:2*radius*pins_per_mm*1j]
    xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
    r = np.hypot(xy[:,0],xy[:,1])
    xy = xy[r<=radius]
    if 'hdiff' in args:
        r = r[r<=radius]
        d =  np.atleast_2d(-args.get('hdiff')*(1/radius*r)**2).T
    else:
        d = np.zeros((xy.shape[0],1))
    if 'center' in args:
        xy = xy + np.array(args.get('center'))
    xy = np.hstack((xy,d))
    return xy


def apply_ramp(trace,**args):
    """Applies on/off ramps to stimulus indentation trace.

    Args:
        trace (array): Indentation trace.

    Kwargs:
        ramp_len (float): length of on/off ramp in s or number of samples (if fs
            not set) (default: 0.05).
        fs (float): sampling frequency (default: None).
        ramp_type (str): 'lin' for linear, 'sin' for sine (default: 'lin').

    Returns:
        Nothing, original trace is modified in place.
    """
    ramp_len = args.get('ramp_len',.05)
    fs = args.get('fs',None)
    ramp_type = args.get('ramp_type','lin')
    if max(ramp_len,0.)==0.:
        return
    if fs is not None:
        ramp_len = round(ramp_len*fs)

    if ramp_type=='lin':    # apply ramp
        trace[:ramp_len] *= np.linspace(0,1,ramp_len)
        trace[-ramp_len:] *= np.linspace(1,0,ramp_len)
    elif ramp_type=='sin' or ramp_type=='sine':
        trace[:ramp_len] *= np.cos(np.linspace(np.pi,2.*np.pi,ramp_len))/2.+.5
        trace[-ramp_len:] *= np.cos(np.linspace(0.,np.pi,ramp_len))/2.+.5
    else:
        raise RuntimeError("ramp_type must be 'lin' or 'sin'")


def apply_pad(trace,**args):
    """Applies zero-padding to stimulus indentation trace.

    Args:
        trace (array): Indentation trace.

    Kwargs:
        pad_len (float): length of on/off ramp in s or number of samples (if fs
            not set) (default: 0.05).
        fs (float): sampling frequency (default: None).

    Returns:
        Padded trace (array).
    """
    pad_len = args.get('pad_len',.05)
    fs = args.get('fs',None)
    if max(pad_len,0.)==0.:
        return
    if fs is not None:
        pad_len = round(pad_len*fs)
    trace = np.concatenate((np.zeros(pad_len),trace,np.zeros(pad_len)))
    return trace
