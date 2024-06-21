# STGS-BMNAS: Straight Through Gumbel Softmax Estimator based Bimodal Neural Architecture Search for Audio-Visual Deepfake Detection
## International Joint Conference on Biometrics(IJCB)-2024 [[paper]](https://arxiv.org/pdf/2406.13384)
# Requriments
    python >=3.6, pytorch==1.5.0 torchvision==0.6.0
# Datasets
 We used two datasets FakeAVCeleb [Link](https://docs.google.com/forms/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform) and SWAN-DF [Link](https://zenodo.org/records/8365616)
# Dataset pre-processing
## Video Preprocessing:
  ### First, run the following script to reshape all RGB videos to 256x256 with 30 fps.
     python3 datasets/prepare_ntu.py --dir=<dir_of_RGB_video>
## Audio pre-processing
  ### Extract all the audio files of the respective video and then create a source.txt file that contains the file name of all audio files in the npy extension and their path.
      python3 preprocess_1.py
  ### Now keep all the videos in one directory and audio files in another directory but the same parent directory and run
      python compute_mean_std.py
  ### Create a “label2.txt” file that contains the video name along with their actual labels.
# Run Experiment:
 ## First, search the hypernets. You can use --parallel for data-parallel. The default setting will require about 128GB of GPU memory, you may adjust the --batch size according to gpu memory capacity.
      python main_darts_searchable_ntu.py –parallel
