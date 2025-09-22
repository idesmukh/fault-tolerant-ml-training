# Fault-tolerant machine learning pipeline training for solar energy forecasting
## Introduction
## Literature review
Fault-tolerant machine learning pipeline training for solar energy forecasting.
## System design
### Requirements gathering
#### Functional requirements

**Training management**
- Load dataset
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
#### System architecture
#### Core components
#### Data flow
#### Failure handling
## System implementation
## System testing
## Results
## Conclusion
## Future work
## References
