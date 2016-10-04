from utils.process_spikes import process_correlation_activity

thresh = .96  # correlation threshold

folder = 'data/correlations-CB'
filename = 'CB_spikes'
n_trials  = 100  # can be up to 100
n_neurons = 10000

process_correlation_activity(folder='%s/neural_activity' % folder,
                             filename=filename,
                             n_neurons=n_neurons,
                             n_trials=n_trials)
