from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
[Fs, x] = audioBasicIO.read_audio_file("./samples/like_themes/track1_mono.wav")

print('Fs: \n', Fs)
print('\n\n\nx: ', len(x))
print('\n\n\nx: ', x[4645760])

F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)

print('F: \n', F)
print('\n\n\nf_names: ', f_names)

plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]) 
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1])
plt.show();