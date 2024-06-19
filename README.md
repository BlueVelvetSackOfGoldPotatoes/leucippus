# Organic Chemistry Game Solver

## Overview

This project aims to develop a machine learning agent capable of solving simple organic chemistry games. The agent will use a combination of language models (LLM), convolutional neural networks (CNN), and deep reinforcement learning (DRL) techniques. The primary goal is to enable the agent to understand and apply chemical rules to generate stable molecules with intended properties.

## Components

### 1. Simulation Environment

The simulation environment will be responsible for generating and visualizing molecular structures as the game progresses. It will use the RDKit library for managing and rendering molecules.

### 2. Agent

The agent will consist of three primary components:

- **Language Model (LLM)**: Generates potential moves based on the current game state using its knowledge of organic chemistry.
- **Convolutional Neural Network (CNN)**: Analyzes images of the current molecular structure to extract features that will inform the DRL model about the current state of the game.
- **Deep Reinforcement Learning (DRL)**: Decides the best move based on the input from the CNN and updates its strategy based on the game's feedback.

### 3. Training Datasets

The training data for this project will be generated in three different formats to accommodate the multimodal approach of the agent:

- **1D (SMILES Strings)**: Simple text representations of molecules.
- **2D Images**: Visual representations of molecules.
- **3D Structures**: Spatial coordinates of molecules in three dimensions.

Each dataset will be paired with natural language descriptions that include not only chemical information but also positional details and functions or chemical properties of the molecule.

## Subplans

### Subplan A: Dataset Creation

#### Objectives
- Generate a comprehensive dataset that includes SMILES strings, 2D and 3D images of molecules, and detailed descriptions.
- Annotate each molecule with functional information and positional details to enrich the training process.

#### Methodology
1. **Molecule Generation**:
   - Use RDKit to generate molecules from base SMILES strings with slight modifications to introduce variations.
2. **Image Rendering**:
   - Generate 2D and 3D representations using RDKit's visualization tools.
3. **Description Writing**:
   - Develop a script to automatically generate detailed descriptions based on the molecule's structure and modifications.
4. **Data Storage**:
   - Save all data in an organized format to facilitate easy access during model training.

### Subplan B: Model Development and Integration

#### Objectives
- Develop and integrate the LLM, CNN, and DRL models to work seamlessly with each other and with the simulation environment.
- Train each model using the generated datasets and refine their performance based on feedback from simulation results.

#### 1. State Representation

- **Convolutional Neural Network (CNN)**: Continues to process images of the molecular structure to extract features, generating a comprehensive state vector that describes the molecule’s current configuration.

#### 2. Enhanced LLM as Move Generator

- **Function**: The LLM generates not only individual moves but also sequences of potential actions based on the game's current state.
- **Input**: Takes the integrated state description, which includes a history of past moves and outcomes, formatted in a structured way to facilitate direct usage by the DRL.
- **Output**: Proposes moves in a structured JSON format, detailing each action's chemical implications and strategic value.

#### 3. Optimized DRL as Decision Maker

- **Function**: The DRL model acts as the decision-making core, evaluating sequences of moves suggested by the LLM. It prioritize actions based on their potential to achieve the game's objectives.
- **Input**: Receives the state vector from the CNN and parsed moves from the LLM.
- **Output**: Selects the optimal sequence of moves or strategic actions to execute.

#### 4. Dynamic Feedback Loop
- **Interaction**: Both the LLM and DRL adjust their strategies based on immediate and long-term game outcomes.

### Subplan C: Evaluation and Refinement

#### Objectives
- Evaluate the overall performance of the agent in the simulation environment.
- Identify areas for improvement and refine the models to enhance their game-solving capabilities.

#### Methodology
1. **Performance Evaluation**:
   - Use a set of metrics to assess the accuracy and efficiency of the agent in making decisions and achieving the desired outcomes.
2. **Refinement**:
   - Based on the evaluation, make necessary adjustments to the models and their integration to optimize performance.
