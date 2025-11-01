import clip 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch 
import torchvision.transforms as T
from PIL import Image 
from liv import load_liv
import moviepy.editor as mp
import os
import tempfile

def extract_frames_from_video(video_path, max_frames=100):
    """Extract frames from video file and return as PIL Images"""
    try:
        # Try using moviepy first
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps
        
        # Calculate frame interval to get roughly max_frames
        total_frames = int(duration * fps)
        if total_frames > max_frames:
            frame_interval = total_frames / max_frames
        else:
            frame_interval = 1
            
        frames = []
        for i in range(min(max_frames, total_frames)):
            time = (i * frame_interval) / fps
            if time >= duration:
                break
            frame = clip.get_frame(time)
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame.astype('uint8'))
            frames.append(pil_image)
        
        clip.close()
        return frames
        
    except Exception as e:
        print(f"Error with moviepy: {e}")
        # Fallback to OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            frame_count += 1
        
        cap.release()
        return frames

# Load model
model = load_liv()
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
transform = T.Compose([T.ToTensor()])

# Video path
video_path = "/Users/ludvigeriksonbrangstrup/Qualia/LIV/fold_napkins.mp4"
task = "fold the napkin"  # You can modify this to describe what's happening in your video

print(f"Extracting frames from {video_path}...")
imgs = extract_frames_from_video(video_path, max_frames=200)
print(f"Extracted {len(imgs)} frames")

if len(imgs) == 0:
    print("No frames extracted from video. Please check the video path.")
    exit(1)

# Convert to tensors
imgs_tensor = []
for img in imgs:
    imgs_tensor.append(transform(img))

imgs_tensor = torch.stack(imgs_tensor)
imgs_tensor = imgs_tensor.to(device)

print("Computing embeddings...")
with torch.no_grad():
    embeddings = model(input=imgs_tensor, modality="vision")
    goal_embedding_img = embeddings[-1]
    token = clip.tokenize([task]).to(device)
    goal_embedding_text = model(input=token, modality="text")
    goal_embedding_text = goal_embedding_text[0] 

print("Computing distances...")
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30,6))
distances_cur_img = []
distances_cur_text = [] 
for t in range(embeddings.shape[0]):
    cur_embedding = embeddings[t]
    cur_distance_img = - model.module.sim(goal_embedding_img, cur_embedding).detach().cpu().numpy()
    cur_distance_text = - model.module.sim(goal_embedding_text, cur_embedding).detach().cpu().numpy()

    distances_cur_img.append(cur_distance_img)
    distances_cur_text.append(cur_distance_text)

distances_cur_img = np.array(distances_cur_img)
distances_cur_text = np.array(distances_cur_text)

# Normalize distances to 0-1 range for comparison plot
distances_img_norm = (distances_cur_img - distances_cur_img.min()) / (distances_cur_img.max() - distances_cur_img.min())
distances_text_norm = (distances_cur_text - distances_cur_text.min()) / (distances_cur_text.max() - distances_cur_text.min())

ax[0].plot(np.arange(len(distances_cur_img)), distances_cur_img, color="tab:blue", label="image", linewidth=3)
ax[0].plot(np.arange(len(distances_cur_text)), distances_cur_text, color="tab:red", label="text", linewidth=3)
ax[1].plot(np.arange(len(distances_cur_img)), distances_cur_img, color="tab:blue", label="image", linewidth=3)
ax[2].plot(np.arange(len(distances_cur_text)), distances_cur_text, color="tab:red", label="text", linewidth=3)
ax[3].plot(np.arange(len(distances_img_norm)), distances_img_norm, color="tab:blue", label="image (normalized)", linewidth=3)
ax[3].plot(np.arange(len(distances_text_norm)), distances_text_norm, color="tab:red", label="text (normalized)", linewidth=3)

ax[0].legend(loc="upper right")
ax[0].set_xlabel("Frame", fontsize=15)
ax[1].set_xlabel("Frame", fontsize=15)
ax[2].set_xlabel("Frame", fontsize=15)
ax[3].set_xlabel("Frame", fontsize=15)
ax[0].set_ylabel("Embedding Distance", fontsize=15)
ax[3].set_ylabel("Normalized Distance (0-1)", fontsize=15)
ax[0].set_title(f"Combined: Image + Language", fontsize=15)
ax[1].set_title("Image Goal Only", fontsize=15)
ax[2].set_title(f"Language Goal: {task}", fontsize=15)
ax[3].set_title("Normalized Comparison", fontsize=15)
ax[3].legend(loc="upper right")
ax[4].imshow(imgs_tensor[-1].permute(1,2,0))
ax[4].set_title("Final Frame (Image Goal)", fontsize=15)
ax[4].axis('off')
asp = 1
ax[0].set_aspect(asp * np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
ax[1].set_aspect(asp * np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
ax[2].set_aspect(asp * np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0])
ax[3].set_aspect(asp * np.diff(ax[3].get_xlim())[0] / np.diff(ax[3].get_ylim())[0])
fig.savefig(f"liv_custom_video.png", bbox_inches='tight')
plt.close()

ax0_xlim = ax[0].get_xlim()
ax0_ylim = ax[0].get_ylim()
ax1_xlim = ax[1].get_xlim()
ax1_ylim = ax[1].get_ylim()
ax2_xlim = ax[2].get_xlim()
ax2_ylim = ax[2].get_ylim()
ax3_xlim = ax[3].get_xlim()
ax3_ylim = ax[3].get_ylim()

def animate(i):
    for ax_subplot in ax:
        ax_subplot.clear()
    ranges = np.arange(len(distances_cur_img))
    if i >= len(distances_cur_img):
        i = len(distances_cur_img) - 1
    line1 = ax[0].plot(ranges[0:i], distances_cur_img[0:i], color="tab:blue", label="image", linewidth=3)
    line2 = ax[0].plot(ranges[0:i], distances_cur_text[0:i], color="tab:red", label="text", linewidth=3)
    line3 = ax[1].plot(ranges[0:i], distances_cur_img[0:i], color="tab:blue", label="image", linewidth=3)
    line4 = ax[2].plot(ranges[0:i], distances_cur_text[0:i], color="tab:red", label="text", linewidth=3)
    line5 = ax[3].plot(ranges[0:i], distances_img_norm[0:i], color="tab:blue", label="image (normalized)", linewidth=3)
    line6 = ax[3].plot(ranges[0:i], distances_text_norm[0:i], color="tab:red", label="text (normalized)", linewidth=3)
    line7 = ax[4].imshow(imgs_tensor[i].permute(1,2,0))
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[1].set_xlabel("Frame", fontsize=15)
    ax[2].set_xlabel("Frame", fontsize=15)
    ax[3].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Embedding Distance", fontsize=15)
    ax[3].set_ylabel("Normalized Distance (0-1)", fontsize=15)
    ax[0].set_title(f"Combined: Image + Language", fontsize=15)
    ax[1].set_title("Image Goal Only", fontsize=15)
    ax[2].set_title(f"Language Goal: {task}", fontsize=15)
    ax[3].set_title("Normalized Comparison", fontsize=15)
    ax[3].legend(loc="upper right")
    ax[4].set_title("Current Frame", fontsize=15)
    ax[4].axis('off')

    ax[0].set_xlim(ax0_xlim)
    ax[0].set_ylim(ax0_ylim)
    ax[1].set_xlim(ax1_xlim)
    ax[1].set_ylim(ax1_ylim)
    ax[2].set_xlim(ax2_xlim)
    ax[2].set_ylim(ax2_ylim)
    ax[3].set_xlim(ax3_xlim)
    ax[3].set_ylim(ax3_ylim)

    return line1, line2, line3, line4, line5, line6, line7

print("Generating animated reward curve...")
# Generate animated reward curve
ani = FuncAnimation(fig, animate, interval=50, repeat=False, frames=len(distances_cur_img)+30)    
ani.save(f"liv_custom_video.gif", dpi=100, writer=PillowWriter(fps=20))

print("Done! Generated files:")
print("- liv_custom_video.png (static plot)")
print("- liv_custom_video.gif (animated reward curve)")