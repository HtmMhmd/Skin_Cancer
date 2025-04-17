# Skin Cancer Detection Project

This project implements a skin cancer detection application using deep learning techniques. The application utilizes a pre-trained UNet model for image segmentation and classification of skin lesions.

## Project Structure

```
skin-cancer-detection
├── app
│   ├── __init__.py
│   ├── inference.py
│   ├── models
│   │   ├── __init__.py
│   │   └── model.py
│   └── utils
│       ├── __init__.py
│       └── preprocessing.py
├── config
│   └── config.yaml
├── data
│   ├── images
│   │   └── example.jpg
│   └── models
│       └── my_checkpoint.pth.tar
├── output
│   └── .gitkeep
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── README.md
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd skin-cancer-detection
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Docker Setup**
   To run the application using Docker, build the Docker image and run the container:
   ```bash
   docker-compose up --build
   ```

## Usage

To run the inference process on a sample image, execute the following command:
```bash
python run.py
```
Make sure to replace `"example.jpg"` in `run.py` with the path to your own image if needed.

## Model Information

The application uses a UNet model architecture for segmentation tasks. The model weights are stored in `data/models/my_checkpoint.pth.tar`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.