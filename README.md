<h1> Bayesian Multivariate Gaussian Random Walk Regression for ERP Estimation </h1>

<p>
	The present analysis implements a multivariate Gaussian random walk (MGRW).  
</p>
<p></p>

<h1> Model </h1>

<p> The present model attempts to estimate event-related potentials (ERPs) across the whole epoch. To that aim, we use an MGRW prior to model the voltage changes across time plus a Gaussian noise parameter. Covariance is assumed as a diagonal matrix of ones. A normal distribution (likelihood) is used for the observed voltages with a half-normal distribution as error parameter. </p>

<p align="center"> &Sigma;<sub>1</sub>,...&Sigma;<sub>C</sub> = E &times; E diagonal matrix </p>
<p align="center"> g<sub>1</sub>,... g<sub>C</sub> ~ GRW(0,1), S &times; E </p>	 
<p align="center"> <em>x</em><sub>1</sub>,... <em>x</em><sub>C</sub> = &Sigma;<sub>c</sub>g<sub>c</sub><sup>T</sup> </p>
<p align="center"> B = S &times; E &times; C matrix of all <em>x</em> </p>
<p align="center"> &alpha;<sub>s</sub> ~ Normal(0, 0.05), S &times; 1 </p>
<p align="center"> &mu; = &alpha;<sub>s</sub> + B<sup>T</sup> </p>
<p align="center"> &sigma; ~ HalfNormal(0.05) + 1 </p>
<p align="center"> y ~ Normal(&mu;, &sigma;) </p>

<p> Where C = 4 mandarin tones (tone 1... tone 4), E = EEG electrodes (32), and S = number of samples (282, 100ms baseline, 1s epoch). Data comes from a tone detection oddball task (Tone 4 was the deviant target, 25% of total stimuli), completed by learners and non-learners of Chinese Mandarin. We fit two models, as described above, to data from each learners and non-learners. </p>

<p> We sampled the model using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 2000 tuning steps, 2000 samples, 4 chains. The model sampled well, with 1.01 > R&#770; > 0.99; BFMIs > 0.9, and bulk ESS > 2000 for all parameters. Ranked trace plots ("trank plots") evidence excellent mixing of chains (see grw_learners/tranks/ and grw_learners/tranks/ folders). </p>

<h1> Results </h1>

<p> The estimates from learners indicate that the target tone (tone 4) induced a strong positive voltage deflection after ~200ms respect to the non-target tones at Pz (i.e., tone 4 induced a P3b). Image below shows the contrasts between tone 4 and each other tone from posterior distributions. </p>

<p align="center">
	<img src="grw_learners/posteriors_learners.png" width="600" height="400" />
</p>

<p> The estimates from non-learners indicate that the target tone (tone 4) induced a milder positive voltage deflection after ~200ms respect to the non-target tones at Pz. Image below shows the contrasts between tone 4 and each other tone from posterior distributions. </p>

<p align="center">
	<img src="grw_non_learners/posteriors_non_learners.png" width="600" height="400" />
</p>


<p> Predictions from the posterior for learners indicate more uncertainty but the P3b is still present. Image below shows contrasts between tone 4 and each other tone from Pz predictions. </p>

<p align="center">
	<img src="grw_learners/predictions_learners.png" width="600" height="400" />
</p>

<p> Predictions from the posterior for non-learners also indicate more uncertainty but there is still a mild P3b. Image below shows contrasts between tone 4 and each other tone from Pz predictions. </p>

<p align="center">
	<img src="grw_non_learners/predictions_non_learners.png" width="600" height="400" />
</p>


<p> Images below show posterior distributions from the learners’ model as scalp topographies (posterior of tone 4 minus all other tones combined). </p>

<p align="center"><strong>5% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_learners/topomap_learners_h5.png" width="600" height="200" /> </p>

<p align="center"><strong>Posterior means</strong></p>
<p align="center"> <img src="grw_learners/topomap_learners_mean.png" width="600" height="200" /> </p>

<p align="center"><strong>95% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_learners/topomap_learners_h95.png" width="600" height="200" /> </p>


<p> Images below show posterior distributions from the non-learners model as scalp topographies (posterior of tone 4 minus all other tones combined). </p>

<p align="center"><strong>5% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_non_learners/topomap_non_learners_h5.png" width="600" height="200" /> </p>

<p align="center"><strong>Posterior means</strong></p>
<p align="center"> <img src="grw_non_learners/topomap_non_learners_mean.png" width="600" height="200" /> </p>

<p align="center"><strong>95% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_non_learners/topomap_non_learners_h95.png" width="600" height="200" /> </p>

<h1> Conclusion </h1>  

<p> The estimates show that there is a difference of P3b amplitude between learners and non-learners, but some uncertainty. The predictions indicate that the models are efficient. However, the current models lack a proper covariance matrix for electrodes, which is relevant to understand voltage variation across the scalp. Further development of these models is required. </p>