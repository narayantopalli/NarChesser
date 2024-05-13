**Credits:**

Move generation: [https://disservin.github.io/chess-library/](url)

Inspiration: AlphaZero - [https://arxiv.org/abs/1712.01815](url), Leela Chess Zero - [https://github.com/LeelaChessZero/](url)


**Instructions:**

-Download CUDA 12.1 and 11.8

-Download Libtorch Debug version

-After Compiling with MSVC place the params.txt in the Debug folder

-Inside the params.txt change the model directory to the directory outside the folders where each of your models are located

-Folder with model should be named current_model, while adding an old_model directory is optional



**Notes:**

-This engine plays at roughly 1500-2000 elo with 60 seconds of thinking time

-Right now it doesn't use a position history for the nnet because of overfitting problems

-All training has so far been done in the cloud through Google Colab

