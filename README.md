# A nonsupervised learning approach for automatic characterization of short-distance boxing training

<p>
<b>Published in: IEEE Transactions on Systems, Man, and Cybernetics: Systems ( Volume: 53, Issue: 11, November 2023) </b>

<p>
<b> Abstract </b> 
<it>Human activity recognition (HAR) for boxing training typically requires specialized hardware or large labeled datasets to identify different types of punches and analyze their basic properties. However, measuring the unpredictability of boxers, a key characteristic to deceive opponents during combats, has remained an unexplored challenge. We tackled this challenge by combining a series of novel techniques and ideas. Unpredictability was computed as the entropy of the trajectories of the Markov chain characterizing the training session, estimated from the combinations (sequences of punches) thrown by the boxer to the training bag. Punch combinations were obtained by detecting punches as outliers in the raw acceleration sensor datastream coming from the bag, and separating them by type with an ensemble of PCA+GMM clusters. Punches were then divided into combinations using statistical analysis of separation intervals. The detection procedure works even with noisy data, in contrast to force-threshold methods commonly reported in the literature. As distinctive features of our system, it uses only unlabeled data from the current training session, and sensor positions and orientations inside the bag are unconstrained. However, punch clustering achieves up to 94% accuracy, comparable to state-of-the-art supervised approaches. These features are currently unseen in the literature and demonstrate the capacity for unsupervised learning techniques to address challenging problems in sports. Multiple validation experiments were consistent with boxers’ level and session performance. Overall, our approach provides significant contributions to the field of HAR and has the potential to improve the way boxing and other sports are trained and evaluated.</it>
## Paper materials



<ul>
  <li> <b>rfidml.ipynb</b> is the jupyter notebook for computing the predictive model, evaluate it and simulate the operation of the smartgate versus a normal gate
    </ul>

Please, cite as: 

<it>J. Vales-Alonso, F. J. González-Castaño, P. López-Matencio and F. Gil-Castiñeira, </it><b>"A Nonsupervised Learning Approach for Automatic Characterization of Short-Distance Boxing Training,"</b> in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 53, no. 11, pp. 7038-7052, Nov. 2023, doi: 10.1109/TSMC.2023.3292146.
