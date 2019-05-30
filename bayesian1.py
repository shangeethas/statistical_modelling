# libraries
import numpy

n = 10.
x = 7.
prior = x / n

# create our data:
traces = {}
for ad_impressions in [10, 100, 1000, 10000]: # maintaining observed CTR of 0.7
    clicks = numpy.array([prior * ad_impressions])    # re-estimate the posterior for
    impressions = numpy.array([ad_impressions])    # increasing numbers of impressions
    with pm.Model() as model:
        theta_prior = pm.Beta('prior', 11.5, 48.5)
        observations = pm.Binomial('obs',n = impressions
                                   , p = theta_prior
                                   , observed = clicks)
        start = pm.find_MAP()
        step = pm.NUTS(state=start)
        trace = pm.sample(5000
                          , step
                          , start=start
                          , progressbar=True)

        traces[ad_impressions] = trace

f, ax = plt.subplots(1)
ax.plot(bins[:-1],prior_counts, alpha = .2)

counts = {}
for ad_impressions in [10, 100, 1000, 10000]:
    trace = traces[ad_impressions]
    posterior_counts, posterior_bins = numpy.histogram(trace['prior'], bins=[j / 100. for j in xrange(100)])
    posterior_counts = posterior_counts / float(len(trace))
    ax.plot(bins[:-1], posterior_counts)
line0, line1, line2, line3, line4 = ax.lines
ax.legend((line0, line1, line2, line3, line4), ('Prior Distribution'
                                                ,'Posterior after 10 Impressions'
                                                , 'Posterior after 100 Impressions'
                                                , 'Posterior after 1000 Impressions'
                                                ,'Posterior after 10000 Impressions'))
ax.set_xlabel("Theta")
ax.axvline(ctr, linestyle = "--", alpha = .5)
ax.grid()
ax.set_ylabel("Probability of Theta")
ax.set_title("Posterior Shifts as Weight of Evidence Increases")
plt.show()
