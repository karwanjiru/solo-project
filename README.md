# AI-Driven Content Generation and Moderation Bot

## Overview

This project provides an AI-driven content generation and moderation system for a social media platform. The system generates engaging content based on user inputs and moderates user-generated content to ensure compliance with community guidelines. It utilizes Generative AI models and moderation algorithms to deliver a seamless experience.
Link to live site(https://huggingface.co/spaces/karwanjiru/solo-project)

## Features

- **Content Generation**: Create posts based on user prompts.
- **Content Moderation**: Detect and filter inappropriate or harmful content.
- **Image Generation**: Create images based on user descriptions.
- **Image Moderation**: Classify and moderate images.
- **Interactive Interface**: User-friendly web interface using Gradio for generating and moderating content.

## Prerequisites
- Storage space

Ensure you have the following installed:

- Python 3.7+ (I used 3.11 though)
- Pip (Python package installer)
- Virtual Environment (optional but recommended)

## Dependencies
***Disclaimer: I'd advice on using the requirements.txt file for exact versions of everything***
    huggingface_hub
    accelerate
    diffusers
    invisible_watermark
    torch
    transformers
    xformers
    torchvision
    Pillow
    gradio
    detoxify
    altair

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/karwanjiru/solo-project
   cd solo-project
   ```

2. **Set Up a Virtual Environment** (Optional but recommended)

   ```bash
   python -m venv venv
   ```
   On Mac use
   ```
   source venv/bin/activate
   ```  
   On Windows use
   ```
   `venv\Scripts\activate`
   ```

3. **Install Dependencies** 
***Exact versions to avoid conflicts***

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Application**

   Run the application using the following command:

   ```bash
   python app.py
   ```
   ***This may also take a min to run ***

   This will launch a Gradio interface in your default web browser.

2. **Access the Web Interface**

   Open your web browser and navigate to `http://localhost:7860` (or the URL provided in the terminal) to access the Gradio interface.

## Usage

### **Content Generation**

- Navigate to the "Generate Post" tab in the Gradio interface.
- Enter a prompt in the "Post Prompt" textbox.
- Adjust parameters such as "Max new tokens," "Temperature," and "Top-p" as needed.
- Click "Generate Post" to create a new post based on your prompt.

### **Content Moderation**

- Navigate to the "Moderate Post" tab.
- Enter the content of the post you want to moderate in the "Post Content" textbox.
- Click "Moderate Post" to check if the content adheres to community guidelines.

### **Image Generation**

- Navigate to the "Generate Image" tab.
- Enter a prompt for the image generation in the "Image Prompt" textbox.
- Click "Generate Image" to create a new image based on your prompt.
        ***This may take a while to run***

### **Image Moderation**

- Navigate to the "Moderate Image" tab.
- Upload an image, use web cam or paste from clipboard for moderation.
- Click "Moderate Image" to check if the image adheres to community guidelines.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your changes. Follow the guidelines outlined in the `CONTRIBUTING.md` file.

## License

This project is licensed under the MIT License. `LICENSE` file for details coming soon.

## Contact

For any questions or support, please contact [Diana Wanjiru](mailto:karwanjiru@gmail.com).

Project still under development....enhancements may be added
---
