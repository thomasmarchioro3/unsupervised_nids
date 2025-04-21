# Unsupervised NIDS models

Collection of unsupervised NIDS models from the existing academic literature, re-implemented in Pytorch.

Current progress:
- [x] KitNET (Kitsune) [Mirsky2018]
- [x] Autoencoder (including stacked and noisy) from [Choi2019]
- [ ] Variational Autoencoder from [Choi2019]
- [x] MemAE [Gong2019]

*Note*: The KitNET implementation uses only NumPy to remain faithful to the original version.

References:

- [Mirsky2018] Mirsky Y, Doitshman T, Elovici Y, Shabtai A. Kitsune: an ensemble of autoencoders for online network intrusion detection. arXiv preprint arXiv:1802.09089. 2018 Feb 25.
- [Choi2019] Choi H, Kim M, Lee G, Kim W. Unsupervised learning approach for network intrusion detection system using autoencoders. The Journal of Supercomputing. 2019 Sep;75:5597-621.
- [Gong2019] Gong D, Liu L, Le V, Saha B, Mansour MR, Venkatesh S, Hengel AV. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. InProceedings of the IEEE/CVF international conference on computer vision 2019 (pp. 1705-1714).