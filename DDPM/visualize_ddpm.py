from PIL import Image
import glob

#use the images in default/samples folder where sample_ddpm stores inference data
frames = []
for i in reversed(range(1000)):
    frames.append(Image.open(f"default/samples/x0_{i}.png"))

# Save as GIF
frames[0].save(
    "output.gif",
    save_all=True,
    append_images=frames[1:],
    duration=10,   # ms per frame (adjust speed)
    loop=0
)
