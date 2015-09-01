import numpy as np

import nengo

def gen_attractor(dimensions=1):
    m = nengo.Network('Attractor')

    # other_t = np.random.random()
    m.goal = [(np.random.random() * .25 + .5) * np.random.choice([-1, 1]) for ii in range(dimensions)]

    with m:

        def in_func(t):
            if t < .25: 
                return [0,]*dimensions
            return m.goal
        in1 = nengo.Node(in_func)

        attractor = nengo.Ensemble(1000, dimensions*2, radius=np.sqrt(dimensions*2))

        for ii in range(dimensions):
            # derivative is difference between state and goal
            nengo.Connection(in1[ii], attractor[ii*2], synapse=.01)
            nengo.Connection(attractor[(ii*2)+1], attractor[ii*2], transform=-1, synapse=.01)

            # state is integrator plus derivative input
            nengo.Connection(attractor[(ii*2)+1], attractor[(ii*2)+1], synapse=.075)
            nengo.Connection(attractor[ii], attractor[(ii*2)+1], synapse=.075)

        # from nengo.utils.functions import whitenoise
        # noise_func = whitenoise(0.1, 10, .001, dimensions=2)
        # def func(t):
        #     return noise_func(t + other_t)
        # noisenode = nengo.Node(output=func)
        #
        # nengo.Connection(noisenode, attractor)
                    
        m.probe_in1 = nengo.Probe(in1)
        m.probe_attractor = nengo.Probe(attractor)
        m.probe_attractor_spikes = nengo.Probe(attractor.neurons)
        # probe_noise = nengo.Probe(noisenode)

    return m

if __name__ == "__main__":

    print 'starting...' 
    for ii in range(8): 

        m = gen_attractor(dimensions=ii+1)

        print 'building model...'
        sim = nengo.Simulator(m)
        for jj in range(50):
            print 'dimensions ', ii, ', trial ', jj
            sim.reset()
            # other_t = np.random.random()
            m.goal = [(np.random.random()) * np.random.choice([-1, 1]) for ii in range(ii+1)]
            sim.run(.75)

            # import matplotlib.pyplot as plt
            #
            # plt.subplot(311)
            # plt.plot(sim.trange(), sim.data[m.probe_in1])
            # plt.legend(['in'])
            # plt.subplot(312)
            # plt.plot(sim.trange(), sim.data[m.probe_attractor])
            # plt.legend(['attractor'])
            # plt.subplot(313)
            # # plt.plot(sim.trange(), sim.data[probe_noise])
            # plt.legend(['noise'])
            # plt.show()

            np.savez_compressed(
                '/home/tdewolf/code/jPCA_ForDistribution/data/attractor/var_explained/%s_%i-dimensional_trial%.3i.dat'%('attractor', ii, jj), 
                array1=np.asarray(sim.data[m.probe_attractor_spikes]))
