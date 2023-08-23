import cv2
import os
from IPython.display import FileLink

vid_fname = 'gans_training.avi'
sample_dir = "samples_from_GAN/"
files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'fake_image' in f]
files.sort()

out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'XVID'), 8, (302,302))
[out.write(cv2.imread(fname)) for fname in files]
out.release()
FileLink('gans_training.avi')