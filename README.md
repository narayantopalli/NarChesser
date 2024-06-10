<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Title Text</title>
    <style>
        .title {
            display: inline-block;
            position: relative;
        }
        .title::after {
            content: "";
            display: block;
            width: 100%;
            height: 2px;
            background-color: black;
            position: absolute;
            left: 0;
            bottom: -5px; /* Adjust the position as needed */
        }
    </style>
</head>
<body>
    <h1 class="title">Title Text</h1>
</body>
</html>

**Credits:**

Move generation: [https://disservin.github.io/chess-library/](url)

Inspiration: AlphaZero - [https://arxiv.org/abs/1712.01815](url), Leela Chess Zero - [https://github.com/LeelaChessZero/](url)


**Instructions:**

-Download CUDA 12.1 and 11.8

-Download Libtorch Debug version and place "libtorch" folder inside the same directory as NarChesser

-After Compiling with MSVC place the params.txt in the Debug folder

-Inside the params.txt change the model directory to the directory outside the folders where each of your models are located

-Folder with model should be named current_model, while adding an old_model directory is optional



**Notes:**

-This engine plays at roughly 1500-2000 elo with 60 seconds of thinking time

-Right now it doesn't use a position history for the nnet because of overfitting problems

-All training has so far been done in the cloud through Google Colab

