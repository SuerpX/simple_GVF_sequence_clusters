import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
def normalization(data):
    new_data = data - data.min(1, keepdim=True)[0]
    new_data /= new_data.max(1, keepdim=True)[0]
    return new_data

def draw_cluster_on_frame(GVF_seq_clusters, frames, clusters_num):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    import sys
    from matplotlib import cm
    from matplotlib.pyplot import imshow
    
    font = ImageFont.truetype("arial.ttf", 20)
    all_added_frames = []
    for num, frame in zip(GVF_seq_clusters, frames):
        im = Image.fromarray(np.uint8(frame)).convert('RGB')

        draw = ImageDraw.Draw(im)
#         print(int(num))
        
        draw.text((0, 0), "cluster:{}/{}".format(int(num), clusters_num), (255, 0, 0), font=font)

        all_added_frames.append(np.asarray(im))
    return all_added_frames

def display_frames_as_gif(frames, path, video_name):
    """
    Displays a list of frames as a gif, with controls
    """
    from matplotlib import animation
    from JSAnimation.IPython_display import display_animation
    from IPython.display import display, HTML
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#     display(display_animation(anim, default_mode='loop'))
    anim.save(path + '/' + video_name, writer=writer)