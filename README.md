# Graph Representation Learning Project

## Overview
This repository contains the Jupyter notebook and associated files for a project on Graph Representation Learning. The project explores various aspects of Graph Neural Networks (GNNs), including random node initialization (RNI), aggregation methods, and the implementation of different GNN architectures like Cooperative Graph Neural Networks (Co-GNNs) and DropGIN.

## Main File
The heart of this project is the Jupyter notebook titled `GraphRepresentationLearning_1084529.ipynb`. It includes comprehensive code and explanations, detailing our experiments, findings, and methodologies.

## Dataset
The project utilizes a reconstructed version of the EXP dataset, adapted to be compatible with the latest updates in Torch Geometric. The dataset is available in pickle and .pt formats and can be found inside the folder Co-GNN. Some results of our models training are stored at the file `arrays.pkl`. The final accuracies are caclulated as the mean of the last 20 epochs of testing. Like this we can have a better idea of how expressive is the net at the end of the learning.

## Additional Resources
- `Co-GNN`: Folder containing additional scripts or resources related to the Cooperative Graph Neural Networks. It is fully based on this repository: 
- `README.md`: The file you are currently reading, offering an overview of the repository's contents.

## Getting Started
To run the notebook:
1. Clone this repository to your local machine or a virtual environment.
2. Ensure you have Jupyter Notebook installed.
3. Install the required dependencies, particularly the latest version of Torch Geometric.
4. Open the `GraphRepresentationLearning_1084529.ipynb` notebook in Jupyter and run the cells.

## Contributing
This project was primarily developed by Candidate number: 1084529. Contributions, suggestions, and discussions are welcome.

## Acknowledgements
This project is built upon insights and foundational work from various repositories and tutorials conducted over the term. Special thanks to the authors and contributors of these resources.
