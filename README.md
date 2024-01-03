<div align="center">

# Few-Shot Object Detection

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Description

In this project, the challenge of detecting objects from new categories without fine-tuning and in conditions of minimal labeled data was addressed.
<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMnBxMDFmajc0MGZvam1yNmE5eHh6ZmdraHltbXRheG9oNmQyY3RreiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/NhUL5T9uuuMDsfDKRV/giphy.gif" />
</p>

To tackle this, a simple yet effective model architecture was proposed. It comprises two main components — a fully convolutional neural network extracting feature descriptions and a transformer-based model associating information from several annotated examples with the extracted representations for making predictions. 

<p align="center">
  <img src="./assets/pipeline.png" />
</p>

The proposed model leverages information from multiple annotated examples and performs one-stage detection. The model was trained and evaluated both on a custom synthetic dataset and FSOD dataset.
<p align="center">
  <img src="./assets/omniglot_prediction.png" height="220" />
  <img src="./assets/fsod_prediction.png" height="220"/>

</p>

## Contributors 
This project was completed by [Stanislav Mikhaylevskiy](https://github.com/lqrhy3) and [Vladimir Chernyavskiy](https://github.com/JJBT). If you have any questions or suggestions regarding this project, please feel free to contact us.
