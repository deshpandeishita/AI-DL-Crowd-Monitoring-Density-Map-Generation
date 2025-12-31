# AI-DeepVision
🧠 Crowd Monitoring & Density Map Generation

A modular, research-grade implementation for exploring, visualizing, and preprocessing the ShanghaiTech Crowd Counting Dataset. This project prepares the dataset for training deep learning models such as CSRNet, MCNN, CANNet, and SANet.

📌 Features

✔ Dataset exploration script

✔ Visualization of crowd images, annotation points, and density maps

✔ Histogram of crowd counts (Part A & Part B)

✔ Clean, modular PyTorch dataset loader

✔ Adaptive & fixed-sigma density map generation

✔ Image preprocessing (resize, normalize, RGB conversion)

✔ Ready for training with any CNN-based crowd counting model

📁 Project Structure
crowd_monitoring/
│
├── config.py
├── explore_dataset.py
├── main.py
│
└── dataset/
    ├── __init__.py
    ├── utils.py
    ├── visualization.py
    └── shanghaitech_dataset.py
│
└── ShanghaiTech/
    ├── part_A/
    └── part_B/

📦 Installation

Clone the repository:

Install required Python libraries:

pip install numpy scipy matplotlib opencv-python torch h5py

📊 Dataset

This repository uses the ShanghaiTech Crowd Counting Dataset, which contains:

Part A — highly congested scenes

Part B — sparse crowd scenes

Each image has a .mat file with annotated person locations.

Density maps are generated using Gaussian kernels.

You must download the dataset manually and place it like this:

crowd_monitoring/ShanghaiTech/
    part_A/
        train_data/
        test_data/
    part_B/
        train_data/
        test_data/

🧪 Usage
🔍 1. Explore the Dataset

Visualize images, density maps, and crowd distribution:

python explore_dataset.py


This will:

Show sample images

Show annotated ground-truth points

Display density maps

Plot histogram of crowd counts

🧵 2. Test the Dataset Pipeline

This checks if PyTorch Dataset + DataLoader works correctly:

python main.py


You should see output like:

Image batch shape: torch.Size([2, 3, 256, 256])
Density batch shape: torch.Size([2, 1, 256, 256])

🧩 Code Components
📌 dataset/utils.py

Loads .mat annotations

Generates adaptive/fixed density maps

Image preprocessing utilities

📌 dataset/shanghaitech_dataset.py

PyTorch Dataset class

Loads images + density maps

Converts to tensors

Ready for training

📌 dataset/visualization.py

Plot images

Plot annotation points

Plot density maps

Plot histograms

📌 config.py

Dataset path

Image size

Density generation mode

📈 Example Output

(Screenshots can be added here after running the scripts)

![Sample Image](assets/sample_image.png)
![Density Map](assets/sample_density.png)
![Histogram](assets/histogram.png)

🚀 Future Work

Add CSRNet training script

Add evaluation + counting error metrics

Upload precomputed density maps

Add real-time crowd counting demo

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.

📄 License

Apache License.

