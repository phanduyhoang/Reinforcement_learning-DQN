# Deep Q-Learning with Convolutional Neural Networks  

This repository implements a **Deep Q-Network (DQN)** using **TensorFlow 2.2+** and **Keras** to train an agent in a simple grid-based environment. The agent learns to navigate a `10x10` grid world, interacting with food and avoiding enemies using reinforcement learning.

## Features  

- Implements a **Deep Q-Network (DQN)** with experience replay  
- Uses **Convolutional Neural Networks (CNNs)** to process visual observations  
- Trains using **TensorFlow 2.2+** and Keras  
- Implements a **custom TensorBoard logger** to track training metrics  
- Supports **epsilon-greedy exploration** with decay  
- Stores **experience replay** in a `deque` buffer  
- Saves trained models when performance exceeds a threshold  

## Installation  

Ensure you have **Python 3.6+** installed. Then, install the required dependencies:  



## How It Works  

- The environment consists of a **player (agent)**, **food (reward)**, and an **enemy (penalty)**.  
- The agent takes **one of 9 possible actions** per step (moving in different directions or staying in place).  
- Rewards are assigned as follows:  
  - **+25** for reaching food  
  - **-300** for hitting an enemy  
  - **-1** for each move (to encourage efficiency)  
- The agent learns via **Q-learning**, updating its CNN-based **Q-function** over episodes.  

## Running the Code  


The agent will train over **20,000 episodes** by default, saving models periodically.

## Model Architecture  

The neural network consists of:  

- **2 Convolutional layers** (256 filters, `3x3` kernel, ReLU, MaxPooling, Dropout)  
- **1 Fully connected layer** (`64` neurons, ReLU)  
- **Output layer** (linear activation for Q-values)  

### Hyperparameters  

| Hyperparameter         | Value  |
|------------------------|--------|
| Discount Factor (γ)    | 0.99   |
| Replay Memory Size     | 50,000 |
| Min Replay Memory      | 1,000  |
| Mini-batch Size        | 64     |
| Target Update Frequency | 5     |
| Epsilon Decay         | 0.99975 |
| Min Epsilon           | 0.001  |
| Episodes              | 20,000 |

## TensorBoard Logging  

Training logs are stored under the `logs/` directory. To visualize the training progress, use:  

tensorboard --logdir logs/

## Model Saving  

The model is saved in the `models/` directory when it achieves a minimum reward threshold (`-200` by default). The saved model filenames follow this format:  

2x256__<max_reward>max_<avg_reward>avg_<min_reward>min__<timestamp>.model

yaml

## Notes  

- TensorFlow 2.2 requires **`tf.random.set_seed(1)`** instead of `tf.set_random_seed(1)`.  
- The `ModifiedTensorBoard` class replaces default TensorBoard behavior for custom logging.  
- Rendering (`SHOW_PREVIEW = True`) significantly slows down training. Disable it unless needed.  

## Future Improvements  

- Implement **Double DQN** for better stability  
- Add **Prioritized Experience Replay**  
- Train using multiple parallel environments  
- Optimize network architecture for faster convergence  

---

This README provides a **complete overview** of the project, making it easier for users to understand, install, run, and modify the code.



