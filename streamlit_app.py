import streamlit as st
import random
import torch
from PIL import Image
from transformers import AutoProcessor, FocalNetForImageClassification
from diffusers import DiffusionPipeline
from detoxify import Detoxify
from huggingface_hub import InferenceClient
import numpy as np
import torchvision.transforms as transforms

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
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
else:
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
    pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max

# Initialize the InferenceClient
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Function to analyze text
def analyze_text(input_text):
    results = Detoxify('original').predict(input_text)
    return results

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
    results = Detoxify('original').predict(post)
    for key, value in results.items():
        if value > 0.5:
            return "Post does not adhere to community guidelines."
    return "Post adheres to community guidelines."

# Function to generate images using the diffusion pipeline
def generate_image(prompt):
    generator = torch.manual_seed(random.randint(0, MAX_SEED))
    image = pipe(prompt=prompt, generator=generator).images[0]
    return image

# Function to moderate images
def moderate_image(image):
    image_tensor = transform(image).unsqueeze(0)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)
    label = model.config.id2label[predicted.item()]
    category = label_to_category.get(label, "Unknown")
    return f"Label: {label}, Category: {category}, Confidence: {confidence.item() * 100:.2f}%"

# Streamlit App Layout

st.title("AI-driven Content Generation and Moderation Bot")
power_device = "GPU" if torch.cuda.is_available() else "CPU"
st.markdown(f"Currently running on {power_device}.")

tabs = ["Chat", "Generate Post", "Moderate Post", "Generate Image", "Moderate Image"]
tab = st.selectbox("Select a tab", tabs)

if tab == "Chat":
    st.header("Chat with the Bot")
    system_message = st.text_input("System message", value="You are a friendly Chatbot meant to assist users in managing social media posts ensuring they meet community guidelines", type="default", disabled=True)
    max_tokens = st.slider("Max new tokens", 1, 2048, 512)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.7)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95)
    
    history = []
    if "history" not in st.session_state:
        st.session_state["history"] = []
    
    message = st.text_input("Your message")
    if st.button("Send"):
        history.append((message, None))
        response = respond(message, st.session_state["history"], system_message, max_tokens, temperature, top_p)
        history[-1] = (message, response)
        st.session_state["history"] = history

    for user_message, bot_message in history:
        st.write(f"**You:** {user_message}")
        st.write(f"**Bot:** {bot_message}")

elif tab == "Generate Post":
    st.header("Generate a Post")
    prompt = st.text_input("Post Prompt")
    max_tokens = st.slider("Max new tokens", 1, 2048, 512)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.7)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95)
    
    if st.button("Generate Post"):
        generated_post = generate_post(prompt, max_tokens, temperature, top_p)
        st.write(f"**Generated Post:** {generated_post}")

elif tab == "Moderate Post":
    st.header("Moderate a Post")
    post_content = st.text_area("Post Content")
    
    if st.button("Moderate Post"):
        moderation_result = moderate_post(post_content)
        st.write(f"**Moderation Result:** {moderation_result}")

elif tab == "Generate Image":
    st.header("Generate an Image")
    image_prompt = st.text_input("Image Prompt")
    
    if st.button("Generate Image"):
        generated_image = generate_image(image_prompt)
        st.image(generated_image, caption="Generated Image")

elif tab == "Moderate Image":
    st.header("Moderate an Image")
    uploaded_image = st.file_uploader("Upload Image for Moderation", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        moderation_result = moderate_image(image)
        st.image(image, caption="Uploaded Image")
        st.write(f"**Classification Result:** {moderation_result}")
