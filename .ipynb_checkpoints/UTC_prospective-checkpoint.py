import nengo
import nengo_spa as spa
import nengolib
import nengo_extras
import pytry
import numpy as np
import pylab
import pandas as pd
import seaborn as sns
import itertools

import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance)
from nengo.utils.ensemble import response_curves, tuning_curves

import os
import sys

class UTC_prospective(pytry.NengoTrial):
    def params(self):
        # Network parameters
        self.param('neuron type', n_type = nengo.Direct())
        self.param('number of neurons per dimension', N=200)
        self.param('minimum of max_rate (uniformly distributed)', min_max_rate = 20)
        self.param('maximum of max_rate (uniformly distributed)', max_max_rate = 50)
        self.param('recurrent time-constant', tau=0.1)
        self.param('control time-constant', tau_control=0.005)
        self.param('number of Legendre Memory dimensions', dimensions=1)
        self.param('base theta', base_theta=1.)
        self.param('effective theta (1/c)', effective_theta=1.0) # set either theta
        self.param('control signal', c=1.0) # or the control signal directly
        self.param('within-trial gain noise standard deviation', within_gain_noise_std=0.0)
        self.param('between-trial gain noise standard deviation', between_gain_noise_std=0.1)
        self.param('within-trial input noise standard deviation', input_noise_std = 0.1)
                
        # Task parameters
        self.param('task type: prospective_motor, prospective_sensory', task_type = 'prospective_motor')
        self.param('full output (True/False)', full_output=False)
        self.param('trial duration', trial_duration=2.0)
        self.param('pre-stimulus interval', prestim = 0.5)
        self.param('post-stimulus interval', poststim = 0.5)

    # define the model
    def model(self, p):
        import sys as system
        np.set_printoptions(threshold=system.maxsize) # ensures all data is saved in .txt
        np.random.seed(p.seed) # ensures that noise generated with numpy differs accross trials
        
        with spa.Network(seed = 0) as model: # fix seed, so neuron tuning curves are consistent accross trials
            # ----------------
            # Hyper-parameters
            # ----------------
            realizer = nengolib.signal.Identity
            solver = nengo.solvers.LstsqL2(reg=1e-3)
            model.config[nengo.Ensemble].neuron_type = p.n_type
            self.neuron_type = model.config[nengo.Ensemble].neuron_type
            
            # ---------------
            # Timing parameters
            # ---------------
            p.thresholds = np.array([0.63155054, 0.59954345, 0.58178892, 0.57019859, 0.56189282, 0.55558088])
            
            # Set control (effective_theta = base_theta/c)
            if p.between_gain_noise_std != 0.:
                p.between_gain_noise = np.random.normal(0, p.between_gain_noise_std)
            else:
                p.between_gain_noise = 0.
            
            if p.effective_theta != None:
                p.c = p.base_theta/p.effective_theta + p.between_gain_noise
            else:
                p.effective_theta = p.base_theta/p.c
            
            def control_function(t):
                return np.random.normal(p.c, p.within_gain_noise_std, 1)
            
            control = nengo.Node(output=control_function)
            gain = nengo.Ensemble(n_neurons=128*p.dimensions, dimensions=1, radius=2)
            
            # Set input (step input, perturbed by noise)
            def stimulus_function(t):
                if t < p.prestim or t > p.trial_duration:
                    return np.random.normal(0, p.input_noise_std, 1)
                else:
                    return np.random.normal(1, p.input_noise_std, 1)
            
            u = nengo.Node(output=stimulus_function)
            
            # if doing motor timing, set maximum time for response
            if p.task_type == 'prospective_motor':
                p.trial_duration = p.effective_theta*3 + p.prestim
            elif p.task_type == 'prospective_sensory':
                p.trial_duration = p.trial_duration + p.prestim

            #-------------------
            def readout(x):
                return x.dot(sys.C.T)
            #-------------------

            # ------------
            # Delay system
            # ------------
            sys = realizer()(
                nengolib.synapses.LegendreDelay(theta=p.base_theta, order=p.dimensions)
            ).realization
            A = sys.A
            B = sys.B
            
            def principle3(t, signals):
                # Euler's method of Principle 3
                # (assumes dt is sufficiently small)
                x, u, gain = signals[:-2], signals[-2], signals[-1]
                return (p.tau*gain) * (A.dot(x) + B.dot(u).squeeze()) + x

            # some mechanism that is responsible for doing the appropriate
            # rescaling of Principle 3, given (x, u, gain)
            mechanism = nengo.Node(size_in=p.dimensions+2, output=principle3)
            mx, mu, mg = mechanism[:-2], mechanism[-2], mechanism[-1]

            # --------------------
            # State representation
            # --------------------
            # there are many different ways you might want to represent x
            # depending on the particular state-space realization and
            # the nonlinear functions you need supported and the range of
            # thetas. all of this affects the encoders, radius, etc.
            x = nengo.Ensemble(
                n_neurons=p.N*p.dimensions,
                dimensions=p.dimensions,
                max_rates = nengo.dists.Uniform(p.min_max_rate, p.max_max_rate),
                #encoders=nengo.dists.Choice(np.eye(dimensions)),
                radius=1.5)
            
            r = nengo.Node(size_in=1)
            
            nengo.Connection(control, gain, synapse=None)
            nengo.Connection(u, mu, synapse=None)
            nengo.Connection(x, mx, synapse=None, solver=solver)
            nengo.Connection(gain, mg, synapse=p.tau_control)
            nengo.Connection(mechanism, x, synapse=p.tau) 
            nengo.Connection(x, r, function=readout)

            # -------------
            # Model outputs
            # -------------
            # probe the post-synaptic current represented by x
            # instead of decoding from x, as this is cheaper
            # and avoids an extra round of filtering

            # State probes
            self.p_x = nengo.Probe(x, synapse=None)
            self.p_r = nengo.Probe(r, synapse=None)
            self.p_u = nengo.Probe(u, synapse=None)
            self.p_gain = nengo.Probe(gain, synapse=None)
            if str(p.n_type) == 'LIF()':
                self.p_x_spikes = nengo.Probe(x.neurons, 'spikes', synapse=None)
                
        return model

    # run the model and evaluate the results
    # When doing motor timing, the trial is ended upon crossing a threshold,
    # and the resulting trial duration is the response time
    # When doing sensory timing, the trial duration is the length of the signal,
    # and the representation at the end of the trial is used in further analysis
    def evaluate(self, p, sim, plt):
        
        sim.run(p.trial_duration + p.poststim)
         
        # Determine estimate for sensory timing, rt for motor timing
        if str(p.n_type) == 'Direct()': # if Direct mode, don't filter
            readout = np.array(sim.data[self.p_r])
            estimate = np.array(sim.data[self.p_r])[-1]
            rt = np.argwhere(readout > p.thresholds[p.dimensions-1])
            if rt.size != 0:
                rt = rt[0][0]
                rt = rt - int(p.prestim*1000)
            else:
                rt = None
        else: # if using neurons, filter readout
            readout = nengo.Alpha(0.05).filt(np.array(sim.data[self.p_r]))
            estimate = readout[int(-p.poststim*1000)]
            rt = np.argwhere(readout > p.thresholds[p.dimensions-1])
            if rt.size != 0:
                rt = rt[0][0]
                rt = rt - int(p.prestim*1000)
            else:
                rt = None
        
        
        if str(p.n_type) == 'LIF()': # if spiking, get probed spikes
            state_spikes = np.array(sim.data[self.p_x_spikes])
        else:
            state_spikes = None
            
        p.n_type = str(p.n_type) # appropriate output
        if p.full_output:
            return dict(time = sim.trange(),
                        threshold = p.thresholds[p.dimensions-1],
                        state = np.array(sim.data[self.p_x]),
                        state_spikes = state_spikes,
                        u = np.array(sim.data[self.p_u]),
                        gain = np.array(sim.data[self.p_gain]),
                        readout = np.array(sim.data[self.p_r]),
                        control = p.c,
                        rt = rt,
                        estimate=estimate[0])
        else:
            return dict(rt = rt,
                        estimate = estimate[0],
                        control = p.c)