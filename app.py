import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoProcessor, FocalNetForImageClassification
import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
from huggingface_hub import InferenceClient
import requests
from io import BytesIO

# Paths and model setup
model_path = "MichalMlodawski/nsfw-image-detection-large"

# Load the model and feature extractor
feature_extractor = AutoProcessor.from_pretrained(model_path)
model = FocalNetForImageClassification.from_pretrained(model_path)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mapping from model labels to NSFW categories
label_to_category = {
    "LABEL_0": "Safe",
    "LABEL_1": "Questionable",
    "LABEL_2": "Unsafe"
}

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the diffusion pipeline
if torch.cuda.is_available():
    torch.cuda.max_memory_allocated(device=device)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
else: 
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
    pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# Initialize the InferenceClient
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Inference function for generating images
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator().manual_seed(seed)
    
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator
    ).images[0] 
    
    return image

# Respond function for the chatbot
def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})
    
    response = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    return response.choices[0].message['content']

# Function to generate posts
def generate_post(prompt, max_tokens, temperature, top_p):
    response = client.chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message['content']

# Function to moderate posts
def moderate_post(post):
    # Implement your post moderation logic here
    if "inappropriate" in post:
        return "Post does not adhere to community guidelines."
    return "Post adheres to community guidelines."

# Function to generate images using the diffusion pipeline
def generate_image(prompt):
    generator = torch.manual_seed(random.randint(0, MAX_SEED))
    image = pipe(prompt=prompt, generator=generator).images[0]
    return image

# Function to moderate images
def moderate_image(image):
    # Convert the PIL image to a format that can be sent for moderation
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    # Replace with your actual image moderation API endpoint
    moderation_api_url = "https://example.com/moderation/api"
    
    # Send the image to the moderation API
    response = requests.post(moderation_api_url, files={"file": image_bytes})
    result = response.json()
    
    # Check the result from the moderation API
    if result.get("moderation_status") == "approved":
        return "Image adheres to community guidelines."
    else:
        return "Image does not adhere to community guidelines."

# Function to classify NSFW images
def classify_nsfw(image):
    image_tensor = transform(image).unsqueeze(0)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)
    label = model.config.id2label[predicted.item()]
    category = label_to_category.get(label, "Unknown")
    return f"Label: {label}, Category: {category}, Confidence: {confidence.item() * 100:.2f}%"

# Create the Gradio interface
css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:
    gr.Markdown("# AI-driven Content Generation and Moderation Bot")
    gr.Markdown(f"Currently running on {power_device}.")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Column():
                chat_interface = gr.ChatInterface(
                    respond,
                    additional_inputs=[
                        gr.Textbox(value="You are a friendly Chatbot meant to assist users in managing social media posts ensuring they meet community guidelines", label="System message", visible=False),
                        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens", visible=False),
                        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature", visible=False),
                        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)", visible=False),
                    ],
                )
                advanced_button = gr.Button("Show Advanced Settings")
                advanced_settings = gr.Column(visible=False)
                with advanced_settings:
                    chat_interface.additional_inputs[0].visible = True
                    chat_interface.additional_inputs[1].visible = True
                    chat_interface.additional_inputs[2].visible = True
                    chat_interface.additional_inputs[3].visible = True
                
                def toggle_advanced_settings():
                    advanced_settings.visible = not advanced_settings.visible
                
                advanced_button.click(toggle_advanced_settings, [], advanced_settings)
        
        with gr.TabItem("Generate Post"):
            post_prompt = gr.Textbox(label="Post Prompt")
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
            generate_button = gr.Button("Generate Post")
            generated_post = gr.Textbox(label="Generated Post")
            generate_button.click(generate_post, [post_prompt, max_tokens, temperature, top_p], generated_post)
        
        with gr.TabItem("Moderate Post"):
            post_content = gr.Textbox(label="Post Content")
            moderate_button = gr.Button("Moderate Post")
            moderation_result = gr.Textbox(label="Moderation Result")
            moderate_button.click(moderate_post, post_content, moderation_result)
        
        with gr.TabItem("Generate Image"):
            image_prompt = gr.Textbox(label="Image Prompt")
            generate_image_button = gr.Button("Generate Image")
            generated_image = gr.Image(label="Generated Image")
            generate_image_button.click(generate_image, image_prompt, generated_image)
        
        with gr.TabItem("Moderate Image"):
            selected_image = gr.Image(type="pil", label="Upload Image for Moderation")
            classify_button = gr.Button("Classify Image")
            classification_result = gr.Textbox(label="Classification Result")
            classify_button.click(classify_nsfw, selected_image, classification_result)

demo.launch()
