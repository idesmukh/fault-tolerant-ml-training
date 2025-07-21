# Fault-tolerant machine learning training pipeline for protein structure prediction
## Introduction
## Literature review
Protein structure prediction is a well-known application of computer science to the protein folding problem. Essentially, the protein folding problem relates to the the construction of a protein's atomic structure using its amino acid sequence [1]. Substantial progress has been made in this regard, with a major one being AlphaFold 2 by Google DeepMind. AlphaFold 2 uses a neural network architecture to predict protein structures with high accuracy [2].

Given the resulting expansion of usage of deep learning in biology and related areas, it is becoming increasingly important to build reliable infrastructure in order to train AI models, and run such models when serving them to clients. This is because deep learning is compute and memory intensive and requires large scale compute infrastructure. DevOps and Site Reliability Engineering (SRE) are crucial components of this objective.

DevOps means Development (Dev) and Operations (Ops). It is aimed at increasing the security, speed, and efficiency of software development and delivery [3]. Site Reliability Engineering (SRE) is a specific implementation of DevOps focused on engineering practices. It involves a focus on engineering, pursuing maximum change velocity, monitoring, emergency response, change management, demand forecasting and capacity planning, provisioning, and efficiency and performance [4].

It is important to ensure that model training is reliable on a large scale. It necessitates the building of tools and processes for scaling. The main objectives should be improving the reliability of training runs and building tools for effective software development practices [5].

The project therefore focuses on a crucial part of  model development, which is training. It involves design, development and implementation of a fault-tolerant ML training pipeline for protein structure prediction. The focus here is on ensuring reliability, rather than the specifics of model performance, such as prediction accuracy.
## System design
### Requirements gathering
#### Functional requirements

**Training management**
- Load protein dataset
- Run a single training job on one GPU node

**Fault-tolerance**
- Save a checkpoint of the model every 15 minutes to persistent storage
- In case of failure, resume training from last checkpoint

**Monitoring**
- Log training progress to console/file
- Show job status (running, completed, failed)

#### Non-functional requirements

**Recovery**
- Training resumes in less than 3 minutes after termination
- Loss of training progress post recovery (from last checkpoint) is zero

**Reliability**
- Recovery rate from failures is 100%

### High-level architecture
#### Core components
#### Data flow
#### Failure handling
## System implementation
## System testing
## Results
## Conclusion
## Future work
## References
[1] Dill, K. A., & MacCallum, J. L. (2012). The protein-folding problem, 50 years on. Science, 338(6110), 1042-1046. https://pmc.ncbi.nlm.nih.gov/articles/PMC2443096/

[2] Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873), 583-589. https://www.nature.com/articles/s41586-021-03819-2

[3] GitLab. (2024). What is DevOps? https://about.gitlab.com/topics/devops/

[4] Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (2016). Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media. https://sre.google/sre-book/introduction/

[5] Isomorphic Labs. (2024). Software Engineer (Reliability Engineering) Job Description. https://job-boards.greenhouse.io/isomorphiclabs/jobs/5538043004