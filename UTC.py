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
import string

import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance)
from nengo.utils.ensemble import response_curves, tuning_curves

import os
import sys

class UTC(pytry.NengoTrial):
    '''
    The UTC class is a NengoTrial type of class and it contains:
    - params: parameters and defaults of the UTC network and the simulation (e.g. type of trial)
    - model: the UTC model, with timing inputs, the recurrent neural network and speed control mechanism
    - evaluate: which allows us to simulate the model given a set of paramers
    '''
    def params(self):
        # Network parameters
        self.param('direct mode (True / False)', direct_mode=False)
        self.param('number of neurons per dimension', N=200)
        self.param('recurrent time-constant', tau=0.1)
        self.param('control time-constant', tau_control=0.005)
        self.param('number of Semantic Pointer dimensions', D=64)
        self.param('number of Legendre Memory dimensions', dimensions=1)
        self.param('base theta', base_theta=1.)
        self.param('effective theta (1/c)', effective_theta=1.0) # set either theta
        self.param('control signal', c=1.0) # or the control signal directly
        self.param('target / window ratio', tw_ratio=1.)
        self.param('within-trial gain noise standard deviation', within_gain_noise_std=0.0)
        self.param('between-trial gain noise standard deviation', 
                   between_gain_noise_std=0.0)
        self.param('within-trial input noise standard deviation', input_noise_std=0.0)
                
        # Task parameters
        self.param('task type: prospective_motor, prospective_sensory, retrospective, gap_procedure', 
                   task_type = 'prospective_motor')
        self.param('gain on task inputs', gain_i=1.)
        self.param('onset of timed stimulus', t_onset=0)
        self.param('offset of timed stimulus', t_offset=1.)
        
        self.param('stimulus function, returns inputs as function of time',
                   stimulus_function=[])
        self.param('number of input stimuli', n_inputs=int(26))
        self.param('input stimuli duration in seconds (array_like 1 x n_input)',
                   input_dur=2.)
        self.param('input stimuli onset (array_like 1 x n_input)', input_onsets=4.)
        self.param('input stimuli inter-stimulus intervals (array_like 1 x n_input)', 
                   input_isi = 0.)
        self.param('vocabulary (defaults to generating random pointers)', vocab=[])
        self.param('store only last x seconds of state vector (eases memory for long simulations)',
                   store_slice = False)
        
        # for gap procedure
        self.param('gap duration', gap_duration=0)
        self.param('gap start, relative to interval onset', gap_start=0)
        self.param('gap similarity (0 for full gap, 1 for no gap)', gap_similarity=0.)
        
        self.param('full output (True/False)', full_output=False)
        self.param('trial duration in seconds', trial_duration=10.)
        self.param('pre-stimulus interval in seconds', prestim=0.)
        self.param('post-stimulus interval in seconds', poststim=0.)
        
        self.param('identifier, based on external variable (e.g. when generating pointers externally)',
                  ID=[])

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
            
            # -----------------
            # Timing parameters
            # -----------------
            # Set noise parameters 
            if p.between_gain_noise_std != 0.:
                p.between_gain_noise = np.random.normal(0, p.between_gain_noise_std)
            else:
                p.between_gain_noise = 0.
                
            # Set control or effective theta (effective_theta = base_theta/c)
            if p.effective_theta != None:
                p.effective_theta = p.effective_theta / p.tw_ratio
                p.c = p.base_theta/p.effective_theta + p.between_gain_noise
            else:
                p.effective_theta = (p.base_theta/p.c) / p.tw_ratio
            
            def control_function(t):
                return np.random.normal(p.c, p.within_gain_noise_std, 1)
            
            # For prospective motor experiments, we need to determine
            # when a behavioral response is generated. We read out
            # the first dimension of the LMU (corresponding to the
            # mean of the temporal window), and when it crosses a threshold
            # a response is generated. The appropriate threshold varies 
            # (slightly) between different orders of the LMU. p.thresholds 
            # contains those appropriate thresholds (to see how they are
            # generated, see prospective_demo.ipynb)
            if (p.task_type=='prospective_motor' or p.task_type == 'gap_procedure'
                or p.task_type == 'distractor_procedure'): 
                p.thresholds = np.array([0.6300808623771766,
                                         0.8821421244327992,
                                         0.9290490234086494,
                                         0.9488498184225395,
                                         0.9597731531689495,
                                         0.9666894084290538,
                                         0.9714564357340751,
                                         0.9749375194262087,
                                         0.9775884384595903,
                                         0.9796725204243686])
                p.threshold = p.thresholds[p.dimensions-1]
                
            else:
                p.threshold = None
                
            control = nengo.Node(output=control_function)
            gain = nengo.Ensemble(n_neurons=200*p.dimensions, dimensions=1, radius=2)
            
            
            # -----------------------------------
            # Input & Semantic Pointer parameters
            # -----------------------------------
            
            # create vocabulary (if none is given as argument)
            if p.vocab == []:
                p.vocab = spa.Vocabulary(p.D,   
                            pointer_gen=spa.vector_generation.OrthonormalVectors(p.D,
                                             rng=np.random.RandomState(1)))
                p.vocab.populate('A;B;C;D;E;F;G;H;I;J;K;L;M;N;O;P;T')
            
            # Set timing input, simple step function
            if p.task_type == 'prospective_sensory' or p.task_type == 'prospective_motor':
                def timing_function(t):
                    if t > p.t_onset and t < p.t_offset and p.task_type!='retrospective':
                        return spa.sym.T
                    else:
                        return 0
                    
            elif p.task_type == 'gap_procedure':
                p.gap_start = p.gap_start + p.prestim
                p.gap_end = p.gap_start + p.gap_duration
                
                def timing_function(t):
                    if t > p.prestim and t < p.gap_start or t > p.gap_end:
                        return spa.sym.T
                    else:
                        return spa.sym.T * p.gap_similarity
            
            elif p.task_type == 'distractor_procedure':
                p.gap_start = p.gap_start + p.prestim
                p.gap_end = p.gap_start + p.gap_duration
                
                def timing_function(t):
                    if t > p.prestim and t < p.gap_start or t > p.gap_end:
                        return spa.sym.T
                    else:
                        return spa.sym.D
            elif p.task_type == 'retrospective':
                def timing_function(t):
                    return 0
               
            timing_input = spa.Transcode(timing_function, output_vocab=p.vocab)
    
            # set stimulus function, sequence of semantic pointers
            # stimulus function can be given as a parameter
            p.letters = [''.join(i) 
                             for i in itertools.product(string.ascii_uppercase, repeat=1)]
            p.letters = p.letters[0:p.n_inputs]
            if p.stimulus_function == []: # if not given, present alphabet
                def stimulus_function(t):
                    idx = int((t // (p.effective_theta / len(p.letters))) % len(p.letters))
                    if t < 1.:
                        return p.letters[idx]
                    else:
                        return 0
                
                stimulus_input = spa.Transcode(stimulus_function, 
                                               output_vocab=p.vocab)
            
            else:
                stimulus_input = spa.Transcode(p.stimulus_function, 
                                               output_vocab=p.vocab)
            
            # superimpose (add) timing input and stimulus input in a state
            state = spa.State(vocab=p.vocab, neurons_per_dimension=200)
            timing_input + p.gain_i * stimulus_input >> state
            
            # if doing motor timing, set maximum time for response
            if p.task_type == 'prospective_motor':
                p.trial_duration = p.effective_theta*3 + p.prestim
            elif p.task_type == 'gap_procedure':
                p.trial_duration = p.effective_theta*2 + p.gap_duration*2 + p.prestim
            elif p.task_type == 'prospective_sensory':
                p.trial_duration = p.trial_duration + p.prestim

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

            # Make scaling mechanisms that are responsible for doing the appropriate
            # rescaling of Principle 3, given (x, u, gain) for each SP dimension
            mechanism = list()
            mx = list()
            mu = list()
            mg = list()
            for d in range(0, p.D):
                m = nengo.Node(size_in=p.dimensions+2, output=principle3)
                mechanism.append(m)                       
                mx.append(m[:-2])
                mu.append(m[-2]) 
                mg.append(m[-1])


            # Create an EnsebleArray
            # with a seperate ensemble for each SP-dimension
            x = nengo.networks.EnsembleArray(
                        n_neurons=p.N * p.dimensions,
                        n_ensembles=p.D,
                        ens_dimensions=p.dimensions,
                        radius=0.3)
            
            #nengo.utils.network.activate_direct_mode(x)
            if p.direct_mode:
                nengo.utils.network.activate_direct_mode(x)
                nengo.utils.network.activate_direct_mode(state)
            
            #-------------------
            def readout(x):
                return x.dot(sys.C.T)
            #-------------------
            r = nengo.Node(size_in=p.D)
            
            nengo.Connection(control, gain, synapse=None)
            for d in range(0,p.D):
                nengo.Connection(state.output[d], mu[d], synapse=None)
                nengo.Connection(x.ea_ensembles[d], mx[d], synapse=None, solver=solver)
                nengo.Connection(gain, mg[d], synapse=None)
                nengo.Connection(mechanism[d], x.ea_ensembles[d], synapse=p.tau)
                nengo.Connection(x.ea_ensembles[d], r[d], function=readout)
            
            # -------------
            # Model outputs
            # -------------
            # probe the post-synaptic current represented by x
            # instead of decoding from x, as this is cheaper
            # and avoids an extra round of filtering

            # probes
            self.p_timing = nengo.Probe(timing_input.output, synapse=None)
            self.p_stimulus = nengo.Probe(stimulus_input.output, synapse=None)
            self.p_state = nengo.Probe(state.output, synapse=None)
            self.p_r = nengo.Probe(r, synapse=None)
            self.p_gain = nengo.Probe(gain, synapse=None)
            
            self.p_x = list()
            for d in range(0,p.D):
                self.p_x.append(nengo.Probe(x.ea_ensembles[d]))
            
                
            
                
        return model

    # run the model and evaluate the results
    # When doing motor timing, the trial is ended upon crossing a threshold,
    # and the resulting trial duration is the response time
    # When doing sensory timing, the trial duration is the length of the signal,
    # and the representation at the end of the trial is used in further analysis
    def evaluate(self, p, sim, plt):
        
        sim.run(p.trial_duration)
        
        p.stimulus_function = [] # overwrite stimulus function for data logging
        
        # Determine estimate for sensory timing, rt for motor timing
        #readout = nengo.Alpha(0.05).filt(np.array(sim.data[self.p_r]))
        #estimate = readout[-1]
        #rt = np.argwhere(readout > p.thresholds[p.dimensions-1])
        #if rt.size != 0:
        #    rt = rt[0][0]
        #    rt = rt - int(p.prestim*1000)
        #else:
        #    rt = None
        
        #state_spikes = np.array(sim.data[self.p_x_spikes])
        

        state = []
        for d in range(0, p.D):
            state.append(sim.data[self.p_x[d]])
        state = np.asarray(state)
        
        if not p.full_output: # compute similarity for non-full output
            state_sim = spa.similarity(sim.data[self.p_state], p.vocab, normalize=False)
            x_sim = spa.similarity(np.swapaxes(state[:,:,0], 0, 1), 
                                                 p.vocab, normalize=False)
            
        if p.store_slice:
            state = state[:, -500:, :] # store only 500ms for analysis (i.e. window readout)
            timing_input = None
            stimulus_input = None
        else:
            timing_input = np.array(sim.data[self.p_timing])
            stimulus_input = np.array(sim.data[self.p_stimulus])
        
        if p.full_output:
            output=dict(time = sim.trange(),
                        state = np.array(sim.data[self.p_state]),
                        x = state,
                        threshold = p.threshold,
                        timing_input = timing_input,
                        stimulus_input = stimulus_input,
                        gain = np.array(sim.data[self.p_gain]),
                        control = p.c,
                        sp_vectors = p.vocab.vectors,
                        sp_keys = list(p.vocab.keys()),
                        #rt = rt,
                        #estimate=estimate[0])
                       )
            p.vocab = [] # overwrite vocab for appropriate output
            return output
        
        elif not p.full_output: # only output similarity to vocab, not full SP
            output=dict(time = sim.trange(),
                        control = p.c,
                        threshold = p.threshold,
                        state = state_sim,
                        x = x_sim)
            p.vocab = []
            return output
        
        
def readout_window(state, D, readout_time, dim, sp):
    rw = nengolib.networks.RollingWindow(n_neurons=100,
                                             theta = 1,
                                             dimensions=dim,
                                             process=None,
                                             legendre=True,
                                             realizer=nengolib.signal.realizers.Identity())
    basis = rw.basis()
    
    Dhistory = []
    for d in range(0, D):
        Dhistory.append(basis * nengo.Alpha(0.05).filt(state)[readout_time, d, :])
    
    sp_coeff = []
    for d in range(0, dim):
        sp_coeff.append(spa.similarity(nengo.Alpha(0.05).filt(state)[readout_time, :, d], 
                                   sp,
                                   normalize=False)[0])
    
    history = np.sum(basis * np.asarray(sp_coeff), axis=1)[::-1]
    
    return history, Dhistory