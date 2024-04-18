# Categorical-Data-Clustering-Parallelization

We study parallelization of categorical data clustering algorithms in an MPI platform. Clustering such data has been a daunting task even for sequential algorithms, mainly due to the challenges in finding suitable similarity/distance measures. We propose a parallel version of the k-modes algorithm, called PV3 (Parallel_V3), which maintains the same clustering quality as produced by the sequential approach while achieving reasonable speed-ups. PV3 is programmed to ensure deterministic processing in a parallel environment.
The othe parallel alogrithm, Paralllel SIG RDM (Parallel_V3_SIG), is a parallel version of the k-modes algorithm with a **non-random initialization technique** called Significane Revised Density Method (SIG RDM). In this version, a method for k-modes initialization called SIG RDM was used and parallelized on top of the original version, PV3.
It is worth mentioning that K-modes often ends up with different clustering results from different sets of initial cluster centers. In other words, it is highly sensitive to the initial cluster centers. So, we needed an initializing algorithm for k-modes to tackle this issue. So, I came up with SIG RDM initialization technique and then parallelized it.
For more detailed explanations, you can refer to : 
https://spectrum.library.concordia.ca/id/eprint/992340/
